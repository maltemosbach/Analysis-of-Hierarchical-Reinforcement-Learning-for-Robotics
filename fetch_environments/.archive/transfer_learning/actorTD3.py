import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import layer

from baselines.her.normalizer import Normalizer


class ActorTD3():

    def __init__(self, sess, env, batch_size, layer_number, FLAGS, hparams, learning_rate=0.001, tau=0.05, policy_freq=2):
        """Actor for the TD3 Actor-Critic implementation.
        Args:
            sess: tensorflow session
            env: environment object containing the Gym envionment
            batch_size (int): size of the training batches
            layer_number (int): number of the layer this actor belongs to
            FLAGS: flags determining how the alogirthm is run
            hparams: hyperparameters set in run.py
            learning_rate (float): learning rate of the actor
            tau (float): polyak averaging coefficient
            policy_freq (int): frequency of policy updates
        """






        self.sess = sess


        self.dimo = env.state_dim
        self.dimg = env.end_goal_dim
        self.dimu = env.subgoal_dim
        self.norm_eps = 0.01
        self.norm_clip = 5
        self.policy_freq = policy_freq

        self.total_it = 0

        # running averages
        with tf.variable_scope('features_stats_actor_1') as vs:
            self.f_stats = Normalizer(self.dimo+self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)



        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset
     
        # Dimensions of action will depend on layer level     
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.actor_name = 'actor_' + str(layer_number) 

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == hparams["layers"] - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim

        self.learning_rate = learning_rate
        # self.exploration_policies = exploration_policies
        self.tau = tau
        self.batch_size = batch_size
        
        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer = self.create_nn(self.features_ph)

        # Target network code "repurposed" from Patrick Emani :^)
        self.weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name]
        # self.num_weights = len(self.weights)
        
        # Create target actor network
        self.target = self.create_nn(self.features_ph, name = self.actor_name + '_target')
        self.target_weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name][len(self.weights):]

        self.update_target_weights = \
        [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                    for i in range(len(self.target_weights))]
    
        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
        self.unnormalized_actor_gradients = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # self.policy_gradient = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.train = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.policy_gradient, self.weights))






    def get_action(self, state, goal):
        actions = self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions

    def get_target_action(self, state, goal):
        actions = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions

    def update(self, state, goal, action_derivs):
        self.total_it += 1
        if self.total_it % self.policy_freq == 0:
            weights, policy_grad, _ = self.sess.run([self.weights, self.policy_gradient, self.train],
                    feed_dict={
                        self.state_ph: state,
                        self.goal_ph: goal,
                        self.action_derivs: action_derivs
                    })

            o = np.asarray(state)
            g = np.asarray(goal)

            #print("o.shape:", o.shape)
            #print("g.shape:", g.shape)
            concat = np.concatenate((state, goal), axis=1)
            #print("concat.shape:", concat.shape)

            self.f_stats.update(concat)
            self.f_stats.recompute_stats()
            #print('stats_f/mean_actor', np.mean(self.sess.run([self.f_stats.mean])))


            return len(weights)

        # self.sess.run(self.update_target_weights)

    # def create_nn(self, state, goal, name='actor'):
    def create_nn(self, features, name=None):

        input = self.f_stats.normalize(features)
        
        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(input, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output