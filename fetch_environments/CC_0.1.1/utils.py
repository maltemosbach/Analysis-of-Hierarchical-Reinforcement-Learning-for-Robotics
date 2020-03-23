import tensorflow as tf
import numpy as np

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


class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v):
        v = v.reshape(-1, self.size)

        self.total_sum += v.sum(axis=0)
        self.total_sumsq += (np.square(v)).sum(axis=0)
        self.total_count[0] += v.shape[0]

        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
        #print("updating normalizer with total count", self.total_count[0])
        #print("new mean:", self.mean)
        #print("new std:", self.std)

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


def get_combinations(combinations):
    hparams = [{}] * len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"])*len(combinations["env"])

    for h in range(len(combinations["env"])):
        for i in range(len(combinations["ac_n"])):
            for j in range(len(combinations["sg_n"])):
                for k in range(len(combinations["replay_k"])):
                    for l in range(len(combinations["layers"])):
                        for m in range(len(combinations["use_target"])):
                            for n in range(len(combinations["sg_test_perc"])):
                                for o in range(len(combinations["use_rb"])):
                                    for p in range(len(combinations["modules"])):
                                        for q in range(len(combinations["run"])):
                                            hparams[h*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"]) + i*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"]) + j*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"]) + k*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"]) + l*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"]) + m*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"]) + n*len(combinations["run"])*len(combinations["modules"])*len(combinations["use_rb"]) + o*len(combinations["run"])*len(combinations["modules"]) + p*len(combinations["run"]) + q ] = {
                                                "env"           : combinations["env"][h],
                                                "ac_n"          : combinations["ac_n"][i],
                                                "sg_n"          : combinations["sg_n"][j],
                                                "replay_k"      : combinations["replay_k"][k],
                                                "layers"        : combinations["layers"][l],
                                                "use_target"    : combinations["use_target"][m],
                                                "sg_test_perc"  : combinations["sg_test_perc"][n],
                                                "use_rb"        : combinations["use_rb"][o],
                                                "modules"       : combinations["modules"][p],
                                                "run"           : combinations["run"][q]

                                                }

    return hparams


