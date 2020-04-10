import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import layer
import time

from baselines.her.normalizer import Normalizer

class CriticTD3():

    def __init__(self, sess, env, layer_number, FLAGS, hparams, learning_rate=0.001, gamma=0.98, tau=0.05, policy_noise=0.2, noise_clip=0.5):

        self.sess = sess

        self.dimo = env.state_dim
        self.dimg = env.end_goal_dim
        self.dimu = env.subgoal_dim
        self.norm_eps = 0.01
        self.norm_clip = 5
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        # running averages
        with tf.variable_scope('features_stats_critic_1') as vs:
            self.f_stats = Normalizer(self.dimo+self.dimg+self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)



        
        self.critic_name = 'critic_' + str(layer_number)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
       
        self.q_limit = -FLAGS.time_scale

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == hparams["layers"] - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim
        self.state_ph = tf.placeholder(tf.float32, shape=(None, env.state_dim), name='state_ph')
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        self.action_ph = tf.placeholder(tf.float32, shape=(None, action_dim), name='action_ph')
        
        self.features_ph = tf.concat([self.state_ph, self.goal_ph, self.action_ph], axis=1)

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        # Create critic network graph
        self.infer_1 = self.create_nn(self.features_ph, name=self.critic_name+"_infer_1")
        self.weights_infer_1 = [v for v in tf.trainable_variables() if self.critic_name+"_infer_1" in v.op.name]
        self.infer_2 = self.create_nn(self.features_ph, name=self.critic_name + "_infer_2")
        self.weights_infer_2 = [v for v in tf.trainable_variables() if self.critic_name+"_infer_2" in v.op.name]

        # Create target critic network graph.  Please note that by default the critic networks are not used and updated.  To use critic networks please follow instructions in the "update" method in this file and the "learn" method in the "layer.py" file.
        self.target_1 = self.create_nn(self.features_ph, name = self.critic_name + '_target_1')
        self.weights_target_1 = [v for v in tf.trainable_variables() if self.critic_name+"_target_1" in v.op.name]
        self.target_2 = self.create_nn(self.features_ph, name = self.critic_name + '_target_2')
        self.weights_target_2 = [v for v in tf.trainable_variables() if self.critic_name+"_target_2" in v.op.name]


        self.update_target_weights_1 = \
        [self.weights_target_1[i].assign(tf.multiply(self.weights_infer_1[i], self.tau) +
                                                  tf.multiply(self.weights_target_1[i], 1. - self.tau))
                    for i in range(len(self.weights_target_1))]

        self.update_target_weights_2 = \
        [self.weights_target_2[i].assign(tf.multiply(self.weights_infer_2[i], self.tau) +
                                                  tf.multiply(self.weights_target_2[i], 1. - self.tau))
                    for i in range(len(self.weights_target_2))]

        self.update_target_weights = tf.group(self.update_target_weights_1, self.update_target_weights_2)
    
        self.wanted_qs = tf.placeholder(tf.float32, shape=(None, 1))

        self.loss = tf.reduce_mean(tf.square(self.wanted_qs - self.infer_1)) + tf.reduce_mean(tf.square(self.wanted_qs - self.infer_2))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.gradient = tf.gradients(self.infer_1, self.action_ph)




    def get_Q_value(self,state, goal, action):
        return self.sess.run(self.infer_1,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]

    def get_target_Q_value(self,state, goal, action):
        return self.sess.run(self.target_1,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]


    def update(self, old_states, old_actions, rewards, new_states, goals, new_actions, is_terminals):

        # Be default, repo does not use target networks.  To use target networks, comment out "wanted_qs" line directly below and uncomment next "wanted_qs" line.  This will let the Bellman update use Q(next state, action) from target Q network instead of the regular Q network.  Make sure you also make the updates specified in the "learn" method in the "layer.py" file.
        '''  
        wanted_qs = self.sess.run(self.infer_1,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })
        '''

        noise = np.clip(np.random.normal(size=old_actions.shape) * self.policy_noise, a_min=-self.noise_clip, a_max=self.noise_clip)

        new_actions = new_actions + noise

        

        wanted_qs_1 = self.sess.run(self.target_1,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })
        #print("wanted_qs_1:", wanted_qs_1)

        wanted_qs_2 = self.sess.run(self.target_2,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })
        #print("wanted_qs_2:", wanted_qs_2)

        wanted_qs = np.minimum(wanted_qs_1, wanted_qs_2)
        #print("wanted_qs:", wanted_qs)
        
       
        for i in range(len(wanted_qs)):
            if is_terminals[i]:
                wanted_qs[i] = rewards[i]
            else:
                wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i][0]

            # Ensure Q target is within bounds [-self.time_limit,0]
            wanted_qs[i] = max(min(wanted_qs[i],0), self.q_limit)
            assert wanted_qs[i] <= 0 and wanted_qs[i] >= self.q_limit, "Q-Value target not within proper bounds"


        self.loss_val, _ = self.sess.run([self.loss, self.train],
                feed_dict={
                    self.state_ph: old_states,
                    self.goal_ph: goals,
                    self.action_ph: old_actions,
                    self.wanted_qs: wanted_qs 
                })

        o = np.asarray(old_states)
        g = np.asarray(goals)
        u = np.asarray(old_actions)

        #print("o.shape:", o.shape)
        #print("g.shape:", g.shape)
        #print("u.shape:", u.shape)
        concat = np.concatenate((old_states, goals, old_actions), axis=1)
        #print("concat.shape:", concat.shape)

        self.f_stats.update(concat)
        self.f_stats.recompute_stats()
        #print('stats_f/mean_critic', np.mean(self.sess.run([self.f_stats.mean])))

        

    def get_gradients(self, state, goal, action):
        grads = self.sess.run(self.gradient,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })

        return grads[0]

    # Function creates the graph for the critic function.  The output uses a sigmoid, which bounds the Q-values to between [-Policy Length, 0].
    def create_nn(self, features, name=None):

        input = self.f_stats.normalize(features)

        if name is None:
            name = self.critic_name        

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, 1, is_output=True)

            # A q_offset is used to give the critic function an optimistic initialization near 0
            output = tf.sigmoid(fc4 + self.q_offset) * self.q_limit

        return output