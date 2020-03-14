import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

def layer(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]
    
    if is_output:
        weight_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        bias_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    else:
        # 1/sqrt(f)
        fan_in_init = 1 / num_prev_neurons ** 0.5
        weight_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)
        bias_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init) 

    weights = tf.compat.v1.get_variable("weights", shape, initializer=weight_init)
    biases = tf.compat.v1.get_variable("biases", [num_next_neurons], initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    relu = tf.nn.relu(dot)
    return relu

def layer_goal_nn(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]
    
    
    fan_in_init = 1 / num_prev_neurons ** 0.5
    weight_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)
    bias_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init) 

    weights = tf.compat.v1.get_variable("weights", shape, initializer=weight_init)
    biases = tf.compat.v1.get_variable("biases", [num_next_neurons], initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    relu = tf.nn.relu(dot)
    return relu


# Below function prints out options and environment specified by user
def print_summary(FLAGS,env):

    print("\n- - - - - - - - - - -")
    print("Task Summary: ","\n")
    print("Environment: ", env.name)
    print("Number of Layers: ", FLAGS.layers)
    print("Time Limit per Layer: ", FLAGS.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", FLAGS.retrain)
    print("Test: ", FLAGS.test)
    print("Visualize: ", FLAGS.show)
    print("Tensorboard: ", FLAGS.tensorboard)
    print("- - - - - - - - - - -", "\n\n")


# Below function ensures environment configurations were properly entered
def check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action):

    # Ensure model file is an ".xml" file
    assert model_name[-4:] == ".xml", "Mujoco model must be an \".xml\" file"

    # Ensure upper bounds of range is >= lower bound of range
    if goal_space_train is not None:
        for i in range(len(goal_space_train)):
            assert goal_space_train[i][1] >= goal_space_train[i][0], "In the training goal space, upper bound must be >= lower bound"

    if goal_space_test is not None:
        for i in range(len(goal_space_test)):
            assert goal_space_test[i][1] >= goal_space_test[i][0], "In the training goal space, upper bound must be >= lower bound"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][0], "In initial state space, upper bound must be >= lower bound"
    
    for i in range(len(subgoal_bounds)):
        assert subgoal_bounds[i][1] >= subgoal_bounds[i][0], "In subgoal space, upper bound must be >= lower bound" 

    # Make sure end goal spaces and thresholds have same first dimension
    if goal_space_train is not None and goal_space_test is not None:
        assert len(goal_space_train) == len(goal_space_test) == len(end_goal_thresholds), "End goal space and thresholds must have same first dimension"

    # Makde sure suboal spaces and thresholds have same dimensions
    assert len(subgoal_bounds) == len(subgoal_thresholds), "Subgoal space and thresholds must have same first dimension"

    # Ensure max action and timesteps_per_action are postive integers
    assert max_actions > 0, "Max actions should be a positive integer"

    assert timesteps_per_action > 0, "Timesteps per action should be a positive integer"


def create_plot(setups, date):

    colors = [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.7, 0.5, 0.85, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]

    x = np.arange(0, setups[0][0].shape[0], 1)
    fig, ax = plt.subplots()

    runs = np.zeros((len(setups), len(setups[0]), len(setups[0][0])), dtype=float)

    for i in range(len(setups)):
        for j in range(len(setups[0])):
            for k in range(len(setups[0][0])):
                runs[i,j,k] = setups[i][j][k]

    # Calculate interquartile range
    intq_range = np.empty((len(setups), len(setups[0][0])), dtype=float)
    for i in range(len(setups)):
        intq_range[i] = iqr(runs[i], axis=0)
    # Calculate average success rate
    average = np.empty((len(setups), len(setups[0][0])), dtype=float)
    for i in range(len(setups)):
        average[i] = np.mean(runs[i], axis=0)

    # Write the success rate array to disk
    with open('sr_data_' + date + '.txt', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(runs.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in runs:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

    yerr = intq_range
    # Creating graphs for the average testing success rate
    for k in range(len(setups)):
        y = average[k]
        yerr = intq_range[k]
        ax.plot(x, average[k], color=colors[k])
        #plt.errorbar(x, y, yerr=yerr)
        plt.fill_between(x, y-yerr, y+yerr, facecolor=colors[k], alpha=0.2)

    plt.title('FetchReach-v1')
    plt.ylabel('Median Test Success Rate')
    ax.set_xlim((0, len(setups[0][0])-1))
    ax.set_ylim((0.0, 1.0))
    plt.xlabel('Epoch')
    plt.grid(True)
    fig.set_size_inches(8, 4)
    Path("./figures").mkdir(parents=True, exist_ok=True)
    shutil.move('sr_data_' + date + '.txt', "./figures")

    plt.savefig("./figures/" + date + ".jpg", dpi=250, facecolor='w', edgecolor='w',
        orientation='landscape',transparent=False, bbox_inches='tight')