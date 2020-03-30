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
    print("Time Limit per Layer: ", FLAGS.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", FLAGS.retrain)
    print("Test: ", FLAGS.test)
    print("Visualize: ", FLAGS.show)
    print("- - - - - - - - - - -", "\n\n")



# Old normalizer class
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
    hparams = [{}] *len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"])*len(combinations["env"])

    for h in range(len(combinations["env"])):
        for i in range(len(combinations["ac_n"])):
            for j in range(len(combinations["sg_n"])):
                for k in range(len(combinations["replay_k"])):
                    for l in range(len(combinations["layers"])):
                        for m in range(len(combinations["use_target"])):
                            for n in range(len(combinations["sg_test_perc"])):
                                for o in range(len(combinations["buffer"])):
                                    for p in range(len(combinations["modules"])):
                                        hparams[h*len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"]) + i*len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"]) + j*len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"]) + k*len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"]) + l*len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"])*len(combinations["use_target"]) + m*len(combinations["modules"])*len(combinations["buffer"])*len(combinations["sg_test_perc"]) + n*len(combinations["modules"])*len(combinations["buffer"]) + o*len(combinations["modules"]) + p] = {
                                            "env"           : combinations["env"][h],
                                            "ac_n"          : combinations["ac_n"][i],
                                            "sg_n"          : combinations["sg_n"][j],
                                            "replay_k"      : combinations["replay_k"][k],
                                            "layers"        : combinations["layers"][l],
                                            "use_target"    : combinations["use_target"][m],
                                            "sg_test_perc"  : combinations["sg_test_perc"][n],
                                            "buffer"        : combinations["buffer"][o],
                                            "modules"       : combinations["modules"][p]

                                            }

    return hparams