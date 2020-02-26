import tensorflow as tf
import numpy as np
from baselines.her.util import store_args, nn
from baselines.her.normalizer import Normalizer

from collections import OrderedDict
from tensorflow.contrib.staging import StagingArea
from baselines import logger

from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)

from baselines.common.mpi_adam import MpiAdam


from baselines.her.replay_buffer import ReplayBuffer


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class ActorAndCritic():

    def __init__(self, sess, writer, env, batch_size, replay_buffer, layer_number, FLAGS, hidden, layers, Q_lr=0.001, pi_lr=0.001, tau=0.05, 
        gamma=0.98, action_l2=1.0, norm_eps=1.0, norm_clip=5, clip_obs=200):

        self.sess = sess
        self.writer = writer
        #self.hparams = hparams
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.hidden = hidden
        self.layers = layers
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer

        self.actor_name = 'actor_' + str(layer_number)
        self.critic_name = 'critic_' + str(layer_number)
        self.Q_lr = Q_lr
        self.pi_lr = pi_lr
        self.gamma = gamma
        self.tau = tau
        self.polyak = 1-tau
        self.clip_return = 50  ## Check on that later!!
        self.action_l2 = action_l2
        self.clip_obs = clip_obs



        self.scope = "layer_" + str(layer_number)



        self.buffer_size = 1000000


        self.ind = 0
        self.i = 0





        self.input_dims = {"o" : 0, "g" : 0, "u" : 0}


        self.state_dim = env.state_dim
        self.input_dims["o"] = env.state_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
            self.input_dims["g"] = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim
            self.input_dims["g"] = env.subgoal_dim

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
            self.max_u = env.action_bounds[0]
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_dim
            self.input_dims["u"] = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim
            self.input_dims["u"] = env.subgoal_dim

        print('self.input_dims:', self.input_dims)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        print("stage_shapes:", stage_shapes)
        self.stage_shapes = stage_shapes


        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(self.input_dims)









    def _create_network(self, input_dims, name=None):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        print('in actorANDcritic.py with self.sess=', self.sess)

        # running averages
        with tf.variable_scope('o_stats') as vs:
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)

        with tf.variable_scope('g_stats') as vs:
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        print("batch_tf in actorANDcritic.py:", batch_tf)
        print("batch in actorANDcritic.py:", batch)



        # Create networks
        with tf.variable_scope('main'):
            self.main = ActorCritic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target'):
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = ActorCritic(
                target_batch_tf, net_type='target', **self.__dict__)
            print('self.target:', self.target)

        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0.)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))


        #If  not training with demonstrations
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run(session=self.sess)
        self._init_target_net()


    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        #self.writer.add_scalar("critic_loss", critic_loss, self.i)
        #self.writer.add_histogram("actor_loss", actor_loss, self.i)
        #print("critic_loss:", critic_loss)
        #print("actor_loss:", actor_loss)
        #print("len(actor_loss):", len(actor_loss))
        self.i += 1
        return critic_loss, actor_loss, Q_grad, pi_grad





    def _vars(self, scope):
        print("In get _vars with scope:", self.scope + '/' + scope)
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        #print("res:", res)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        #print('For actorANDcritic.py in _global_vars with res:', res)
        return res


    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs


    def train(self, stage=True):
        #print('train() in actorANDcritic.py is called!')
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss


    def clear_buffer(self):
        self.buffer.clear_buffer()




    ## Get batch from original experience buffer in the form of the HER replay buffer!!!!!!!!!!!!!!!!!!!!!
    def sample_batch(self, update_stats=True):
        if self.replay_buffer.size >= self.batch_size:
            old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()


            transitions = {}
            transitions["g"] = goals
            transitions["o"] = old_states
            transitions["u"] = actions
            transitions["o_2"] = new_states
            transitions["g_2"] = goals
            transitions["r"] = rewards

            o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
            transitions["o"], transitions["g"] = self._preprocess_og(o, g)
            transitions["o_2"], transitions["g_2"] = self._preprocess_og(o_2, g)
            #transitions_batch = [goals, old_states, actions, new_states, goals, rewards]


            if update_stats:
                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])

                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()






            transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
            '''

            f= open("/Users/maltemosbach/Desktop/transitions_hac.txt","a+")
            #f.write("self.buffer.get_current_size(): " + str(self.buffer.get_current_size()) + "\n")
            f.write("self.stage_shapes.keys(): " + str(self.stage_shapes.keys()) + "\n")
            f.write("Batch no. " + str(self.ind) + " \n")
            f.write("Batch size: " + str(self.batch_size) + " \n")
            f.write("str(len(transitions['g'])): " + str(len(transitions['g'])) + " \n")
            f.write("transitions['g']: " + str(transitions['g']) + "\n")
            f.write("str(len(transitions['o'])): " + str(len(transitions['o'])) + " \n")
            f.write("transitions['o']: " + str(transitions['o']) + "\n")
            f.write("str(len(transitions['u'])): " + str(len(transitions['u'])) + " \n")
            f.write("transitions['u']: " + str(transitions['u']) + "\n")
            f.write("str(len(transitions['o_2'])): " + str(len(transitions['o_2'])) + " \n")
            f.write("transitions['o_2']: " + str(transitions['o_2']) + "\n")
            f.write("str(len(transitions['g_2'])): " + str(len(transitions['g_2'])) + " \n")
            f.write("transitions['g_2']: " + str(transitions['g_2']) + "\n")
            f.write("str(len(transitions['r'])): " + str(len(transitions['r'])) + " \n")
            f.write("transitions['r']: " + str(transitions['r']) + "\n")
            '''

            self.ind += 1


        else:
            assert False, "Sample_batch should only be called with enough transitions in the replay_buffer"


        # Batch should have the form [g, o, u, o_2, g_2, r]


        

        return transitions_batch


    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))


    def get_actions(self, o, g, use_target_net=False,
                    compute_Q=False):

        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        #print('self.logs():', self.logs())

        while len(ret) == 1:
            ret = ret[0]

        return ret



    def get_Q_values(self, o, g, u, use_target_net=False):

        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.Q_pi_tf]

        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _preprocess_og(self, o, g):
            o = np.clip(o, -self.clip_obs, self.clip_obs)
            g = np.clip(g, -self.clip_obs, self.clip_obs)
            return o, g






class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)






