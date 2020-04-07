
import numpy as np
from experience_buffer import ExperienceBuffer
from transitions_buffer import TransitionsBuffer
import time
import tensorflow as tf
import os

from ddpg import DDPG
from ddpgL1 import DDPGL1
from actor import Actor
from critic import Critic
from criticTD3 import CriticTD3
from actorTD3 import ActorTD3

from baselines.her.util import convert_episode_to_batch_major



def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class Layer():
    def __init__(self, layer_number, FLAGS, env, sess, writer, agent_params, hparams):
        """The new layer class for HAC
        Args:
            layer_number (int): number of this layer with 0 being the lowest layer
            FLAGS: flags for configuring the running of the algorithm
            env: environment object
            sess: TensorFlow session
            writer: summary writer
            agent_params: parameters of the agent
            hparams: hyperparameters from initialize HAC
        """


        self.layer_number = layer_number
        self.FLAGS = FLAGS
        self.sess = sess
        self.hparams = hparams
        self.writer = writer


        self.dims = {"g" : 0, "o" : 0, "u" : 0}
        self.dims["o"] = env.state_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == hparams["layers"] - 1:
            self.dims["g"] = env.end_goal_dim
        else:
            self.dims["g"] = env.subgoal_dim

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.dims["u"] = env.action_dim
        else:
            self.dims["u"] = env.subgoal_dim


        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if hparams["layers"] > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions

        self.T = self.time_limit
        self.dimo = self.dims['o']
        self.dimg = self.dims['g']
        self.dimu = self.dims['u']

        self.current_state = None
        self.goal = None

        self.count_ind = 0

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**6


        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = hparams["replay_k"]

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)



        # Configure buffers
        self.buffer_size = self.buffer_size_ceiling

        self.batch_size = 256
        self.experience_buffer = ExperienceBuffer(self.buffer_size, self.batch_size)

        # info is not needed for reward function
        def reward_fun(ag_2, g):  # vectorized
            return env.gymEnv.compute_reward(achieved_goal=ag_2, desired_goal=g, info=0)

        transitions_buffer_shapes = {'o': self.dimo, 'g': self.dimg, 'u': self.dimu, 'r': 1, 'o_2': self.dimo}
        self.transitions_buffer = TransitionsBuffer(transitions_buffer_shapes, self.buffer_size, hparams["replay_k"], reward_fun, sampling_strategy=hparams["samp_str"])




        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        self.hidden = 256
        self.layers = 3

        self.action_l2 = None
        if self.layer_number == 0:
            self.action_l2 = 1.0
            self.noise_perc = [hparams["ac_n"] for i in range(4)]
        else:
            self.action_l2 = 0.0
            self.noise_perc = [hparams["sg_n"] for i in range(3)]

        assert hparams["modules"][0] == "ddpg", "Lowest layer should be a ddpg module"


        if hparams["modules"][self.layer_number] == "ddpg":
            self.policy = DDPG(self.sess, env, hparams, self.batch_size, self.experience_buffer, self.transitions_buffer, self.layer_number, FLAGS, self.hidden, self.layers, self.time_limit, buffer_type=hparams["buffer"][self.layer_number], action_l2=self.action_l2)
            self.critic = None
            self.actor = None
        elif hparams["modules"][self.layer_number] == "actorcritic":
            self.critic = Critic(sess, env, self.layer_number, FLAGS, hparams)
            self.actor = Actor(sess, env, self.batch_size, self.layer_number, FLAGS, hparams)
            self.policy = None
        elif hparams["modules"][self.layer_number] == "ddpgL1":
            self.policy = DDPGL1(self.sess, env, hparams, self.batch_size, self.experience_buffer, self.transitions_buffer, self.layer_number, FLAGS, self.hidden, self.layers, self.time_limit, buffer_type=hparams["buffer"][self.layer_number], action_l2=self.action_l2)
            self.critic = None
            self.actor = None
        elif hparams["modules"][self.layer_number] == "TD3":
            self.critic = CriticTD3(sess, env, self.layer_number, FLAGS, hparams)
            self.actor = ActorTD3(sess, env, self.batch_size, self.layer_number, FLAGS, hparams)
            self.policy = None
        else:
            assert False        

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False

        self.subgoal_penalty = agent_params["subgoal_penalty"]



    # Add noise to provided action
    def add_noise(self,action, env):

        # Noise added will be percentage of range
        if self.layer_number == 0:
            action_bounds = env.action_bounds
            action_offset = env.action_offset
        else:
            action_bounds = env.subgoal_bounds_symmetric
            action_offset = env.subgoal_bounds_offset

        assert len(action) == len(action_bounds), "Action bounds must have same dimension as action"
        assert len(action) == len(self.noise_perc), "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        for i in range(len(action)):
            action[i] += np.random.normal(0,self.noise_perc[i] * action_bounds[i])

            action[i] = max(min(action[i], action_bounds[i]+action_offset[i]), -action_bounds[i]+action_offset[i])

        return action


    # Select random action
    def get_random_action(self, env):

        if self.layer_number == 0:
            action = np.zeros((env.action_dim))
        else:
            action = np.zeros((env.subgoal_dim))

        # Each dimension of random action should take some value in the dimension's range
        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] = np.random.uniform(-env.action_bounds[i] + env.action_offset[i], env.action_bounds[i] + env.action_offset[i])
            else:
                action[i] = np.random.uniform(env.subgoal_bounds[i][0],env.subgoal_bounds[i][1])

        return action


    # Function selects action using an epsilon-greedy policy
    def choose_action(self,agent, env, subgoal_test):

        # If testing mode or testing subgoals, action is output of actor network without noise
        o = self.current_state
        g = self.goal
        if agent.FLAGS.test or subgoal_test:
            if self.hparams["modules"][self.layer_number] == "ddpg" or self.hparams["modules"][self.layer_number] == "ddpgL1":
                return self.policy.get_actions(o, g, use_target_net=self.hparams["use_target"][self.layer_number]), "Policy", subgoal_test
            elif self.hparams["modules"][self.layer_number] == "actorcritic" or self.hparams["modules"][self.layer_number] == "TD3":
                if self.hparams["use_target"][self.layer_number]:
                    return self.actor.get_target_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0], "Policy", subgoal_test
                else:
                    return self.actor.get_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0], "Policy", subgoal_test
            else:
                assert False
            
        else:

            if np.random.random_sample() > 0.3:
                # Choose noisy action
                if self.hparams["modules"][self.layer_number] == "ddpg" or self.hparams["modules"][self.layer_number] == "ddpgL1":
                    action = self.add_noise(self.policy.get_actions(o, g, use_target_net=self.hparams["use_target"][self.layer_number]), env)
                elif self.hparams["modules"][self.layer_number] == "actorcritic" or self.hparams["modules"][self.layer_number] == "TD3":
                    if self.hparams["use_target"][self.layer_number]:
                        action = self.add_noise(self.actor.get_target_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0],env)
                    else:
                        action = self.add_noise(self.actor.get_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0],env)
                else:
                    assert False

                action_type = "Noisy Policy"

            # Otherwise, choose random action
            else:
                action = self.get_random_action(env)

                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < agent.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False



            return action, action_type, next_subgoal_test


    # Create action replay transition by evaluating hindsight action given original goal
    def perform_action_replay(self, hindsight_action, next_state, goal_status):

        # Determine reward (0 if goal achieved, -1 otherwise) and finished boolean.  The finished boolean is used for determining the target for Q-value updates
        if goal_status[self.layer_number]:
            reward = 0
            finished = True
        else:
            reward = -1
            finished = False

        # Transition will take the form [old state, hindsight_action, reward, next_state, goal, terminate boolean, None]
        transition = [self.current_state, hindsight_action, reward, next_state, self.goal, finished, None]

        if self.FLAGS.all_trans or self.FLAGS.hind_action:
            print("\nLevel %d Hindsight Action: " % self.layer_number, transition)

        # Add action replay transition to layer's replay buffer
        self.experience_buffer.add(np.copy(transition))


    # Create initial goal replay transitions
    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, env, total_layers):

        # Create transition evaluating hindsight action for some goal to be determined in future.  Goal will be ultimately be selected from states layer has traversed through.  Transition will be in the form [old state, hindsight action, reward = None, next state, goal = None, finished = None, next state projeted to subgoal/end goal space]

        if self.layer_number == total_layers - 1:
            hindsight_goal = env.project_state_to_end_goal(env.sim, next_state)
        else:
            hindsight_goal = env.project_state_to_subgoal(env.sim, next_state)

        transition = [self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal]

        if self.FLAGS.all_trans or self.FLAGS.prelim_HER:
            print("\nLevel %d Prelim HER: " % self.layer_number, transition)

        self.temp_goal_replay_storage.append(np.copy(transition))

        """
        # Designer can create some additional goal replay transitions.  For instance, higher level transitions can be replayed with the subgoal achieved in hindsight as the original goal.
        if self.layer_number > 0:
            transition_b = [self.current_state, hindsight_action, 0, next_state, hindsight_goal, True, None]
            # print("\nGoal Replay B: ", transition_b)
            self.experience_buffer.add(np.copy(transition_b))
        """





    # Return reward given provided goal and goal achieved in hindsight
    def get_reward(self,new_goal, hindsight_goal, goal_thresholds):

        assert len(new_goal) == len(hindsight_goal) == len(goal_thresholds), "Goal, hindsight goal, and goal thresholds do not have same dimensions"

        if goal_distance(new_goal, hindsight_goal) > goal_thresholds[0]:
            return -1

        # Else goal is achieved
        return 0



    # Finalize goal replay by filling in goal, reward, and finished boolean for the preliminary goal replay transitions created before
    def finalize_goal_replay(self,goal_thresholds):

        # Choose transitions to serve as goals during goal replay.  The last transition will always be used
        num_trans = len(self.temp_goal_replay_storage)

        num_replay_goals = self.num_replay_goals
        # If fewer transitions that ordinary number of replay goals, lower number of replay goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans


        if self.FLAGS.all_trans or self.FLAGS.HER:
            print("\n\nPerforming Goal Replay for Level %d\n\n" % self.layer_number)
            print("Num Trans: ", num_trans, ", Num Replay Goals: ", num_replay_goals)


        indices = np.zeros((num_replay_goals))
        indices[:num_replay_goals-1] = np.random.randint(num_trans,size=num_replay_goals-1)
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)

        if self.FLAGS.all_trans or self.FLAGS.HER:
            print("Selected Indices: ", indices)

        # For each selected transition, update the goal dimension of the selected transition and all prior transitions by using the next state of the selected transition as the new goal.  Given new goal, update the reward and finished boolean as well.
        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)

            if self.FLAGS.all_trans or self.FLAGS.HER:
                print("GR Iteration: %d, Index %d" % (i, indices[i]))

            new_goal = trans_copy[int(indices[i])][6]
            # for index in range(int(indices[i])+1):
            for index in range(num_trans):
                # Update goal to new goal
                trans_copy[index][4] = new_goal

                # Update reward
                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds)

                # Update finished boolean based on reward
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                # Add finished transition to replay buffer
                if self.FLAGS.all_trans or self.FLAGS.HER:
                    print("\nNew Goal: ", new_goal)
                    print("Upd Trans %d: " % index, trans_copy[index])

                self.experience_buffer.add(trans_copy[index])


        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []





    # Create transition penalizing subgoal if necessary.  The target Q-value when this transition is used will ignore next state as the finished boolena = True.  Change the finished boolean to False, if you would like the subgoal penalty to depend on the next state.
    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):

        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None]

        if self.FLAGS.all_trans or self.FLAGS.penalty:
            print("Level %d Penalty Trans: " % self.layer_number, transition)

        self.experience_buffer.add(np.copy(transition))



    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True
                
        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < self.hparams["layers"]-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False





    def append_transition_to_episode(self, env, o, u, ag, g, obs, achieved_goals, acts, goals, info_is_sgtt, layer_number, goal_status, highest_layer, is_sgtt):

        o_new = env.obs['observation']
        if layer_number == highest_layer:
            ag_new = env.project_state_to_end_goal(env.sim, o_new)
        else:
            ag_new = env.project_state_to_subgoal(env.sim, o_new)


        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        acts.append(u.copy())
        goals.append([g.copy()])
        o[...] = o_new
        ag[...] = ag_new

        if layer_number > 0:
            info_is_sgtt.append(np.array([[is_sgtt]]))






    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, env, subgoal_test = False, episode_num = None):

        #print("\nTraining Layer %d" % self.layer_number)

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and agent.FLAGS.show and agent.hparams["layers"] > 1:
            env.display_subgoals(agent.goal_array)
            # env.sim.data.mocap_pos[3] = env.project_state_to_end_goal(env.sim,self.current_state)
            # print("Subgoal Pos: ", env.sim.data.mocap_pos[1])

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0


        # notations for the replay buffer
        o = np.empty((1, self.dims['o']), np.float32)   # observations
        ag = np.empty((1, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.current_state
        self.g = self.goal
        o_new = np.empty((1, self.dims['o']))
        ag_new = np.empty((1, self.dims['g']))

        if self.layer_number == self.hparams["layers"]-1:
            ag[:] = env.project_state_to_end_goal(env.sim, self.current_state)
        else:
            ag[:] = env.project_state_to_subgoal(env.sim, self.current_state)

        self.T = self.time_limit

        # generate episodes
        obs, achieved_goals, acts, goals, info_is_sgtt = [], [], [], [], []



        while True:

            #print("len(self.acts) for layer", self.layer_number, ":", len(self.acts))
            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(agent, env, subgoal_test)




            if self.FLAGS.Q_values:
                # print("\nLayer %d Action: " % self.layer_number, action)
                if layer_number == 0:
                    print("Layer %d Q-Value: " % self.layer_number, self.policy.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(action,(1,len(action)))))
                else:
                    print("Layer %d Q-Value: " % self.layer_number, self.critic.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(action,(1,len(action)))))
                if self.layer_number == 2:
                    test_action = np.copy(action)
                    test_action[:3] = self.goal
                    print("Layer %d Goal Q-Value: " % self.layer_number, self.policy.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(test_action,(1,len(test_action)))))


            # If next layer is not bottom level, propose subgoal for next layer to achieve and determine whether that subgoal should be tested
            if self.layer_number > 0:

                agent.goal_array[self.layer_number - 1] = action

                goal_status, max_lay_achieved = agent.layers[self.layer_number - 1].train(agent, env, next_subgoal_test, episode_num)

            # If layer is bottom level, execute low-level action
            else:
                next_state = env.execute_action(action)

                # Increment steps taken
                agent.steps_taken += 1
                #print("Num Actions Taken: ", agent.steps_taken)

                if agent.steps_taken >= env.max_actions:
                    print("Out of actions (Steps: %d)" % agent.steps_taken)

                agent.current_state = next_state

                # Determine whether any of the goals from any layer was achieved and, if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

            attempts_made += 1
            #print(attempts_made, "attempts made by layer", self.layer_number)

            # Print if goal from current layer as been achieved
            if goal_status[self.layer_number]:
                if self.layer_number < agent.hparams["layers"] - 1:
                    print("SUBGOAL ACHIEVED")
                print("\nEpisode %d, Layer %d, Attempt %d Goal Achieved" % (episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)
                if self.layer_number == agent.hparams["layers"] - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = env.project_state_to_subgoal(env.sim, agent.current_state)

            # hindsight action for replay buffer
            hindsight_u = hindsight_action
            if hindsight_u.ndim == 1:
                hindsight_u = hindsight_u.reshape(1, -1)


            # Next, create hindsight transitions if not testing
            if not agent.FLAGS.test:

                # Appending transition to the episode
                # It is a subgoal testing transition
                if next_subgoal_test:
                    # The subgoal has not been achieved
                    if agent.layers[self.layer_number-1].maxed_out:
                        self.append_transition_to_episode(env, o, hindsight_u, ag, self.g, obs, achieved_goals, acts, goals, info_is_sgtt, self.layer_number, goal_status, self.hparams["layers"]-1, is_sgtt=-1)
                    # The subgoal has been achieved
                    else:
                        self.append_transition_to_episode(env, o, hindsight_u, ag, self.g, obs, achieved_goals, acts, goals, info_is_sgtt, self.layer_number, goal_status, self.hparams["layers"]-1, is_sgtt=1)
                # It is not a subgoal testing transition
                else:
                    self.append_transition_to_episode(env, o, hindsight_u, ag, self.g, obs, achieved_goals, acts, goals, info_is_sgtt, self.layer_number, goal_status, self.hparams["layers"]-1, is_sgtt=0)





                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.hparams["layers"])


                # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])
                    self.transitions_buffer.penalize_subgoal({'u': action, 'o': self.current_state, 'o_2': agent.current_state, 'r': self.subgoal_penalty, 'is_t': True, 'g': self.goal})


            # Print summary of transition
            if agent.FLAGS.verbose:

                print("\nEpisode %d, Level %d, Attempt %d" % (episode_num, self.layer_number,attempts_made))
                # print("Goal Array: ", agent.goal_array, "Max Lay Achieved: ", max_lay_achieved)
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)
                if self.layer_number == agent.hparams["layers"] - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))
                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)



            # Update state of current layer
            self.current_state = agent.current_state

            # Return to previous level to receive next subgoal if applicable
            # if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:

                if self.layer_number == agent.hparams["layers"]-1:
                    print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)



                # If not testing, finish goal replay by filling in missing goal and reward values before returning to prior level.
                if not agent.FLAGS.test:
                    if self.layer_number == agent.hparams["layers"] - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.subgoal_thresholds

                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):

                    if not agent.FLAGS.test and self.hparams["buffer"][self.layer_number] == "transitions":
                        obs.append(o.copy())
                        achieved_goals.append(ag.copy())
                        if self.layer_number == 0:
                            episode = dict(o=obs,
                                   u=acts,
                                   g=goals,
                                   ag=achieved_goals)
                        else:
                            #print("Creating episode for layer 1:")
                            #print("acts:", acts)
                            #print("info_is_sgtt:", info_is_sgtt)
                            episode = dict(o=obs,
                                   u=acts,
                                   g=goals,
                                   ag=achieved_goals,
                                   is_sgtt=info_is_sgtt)
                        episode = convert_episode_to_batch_major(episode)

                        self.transitions_buffer.store_episode(episode, goal_status[self.layer_number])

                    return goal_status, max_lay_achieved



    # Update actor and critic networks
    def learn(self, num_updates):

        if self.hparams["modules"][self.layer_number] == "ddpg" or self.hparams["modules"][self.layer_number] == "ddpgL1":
            print("learning layer {layer_number} ({module}) with {buffer} buffer".format(layer_number=self.layer_number, module=self.hparams["modules"][self.layer_number], buffer=self.hparams["buffer"][self.layer_number]))
            # Update nets if replay buffer is large enough
            if self.experience_buffer.size >= self.batch_size or self.hparams["buffer"][self.layer_number] != "experience":
                # Update main nets num_updates times
                for _ in range(num_updates):
                    self.policy.train()

                # Update all target nets
                self.policy.update_target_net()

        elif self.hparams["modules"][self.layer_number] == "actorcritic" or self.hparams["modules"][self.layer_number] == "TD3":
            print("learning layer {layer_number} ({module}) with {buffer} buffer".format(layer_number=self.layer_number, module=self.hparams["modules"][self.layer_number], buffer=self.hparams["buffer"][self.layer_number]))
            if self.hparams["buffer"][self.layer_number] == "experience":
                if self.hparams["use_target"][self.layer_number]:
                    for _ in range(num_updates):
                        # Update weights of non-target networks
                        if self.experience_buffer.size > 250:
                            old_states, actions, rewards, new_states, goals, is_terminals = self.experience_buffer.get_batch()

                            self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_target_action(new_states,goals), is_terminals)
                            action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                            self.actor.update(old_states, goals, action_derivs)
                    # Update weights of target networks
                    self.sess.run(self.critic.update_target_weights)
                    self.sess.run(self.actor.update_target_weights)

                else:
                    for _ in range(num_updates):
                        # Update weights of non-target networks
                        if self.experience_buffer.size > 250:
                            old_states, actions, rewards, new_states, goals, is_terminals = self.experience_buffer.get_batch()

                            self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_action(new_states,goals), is_terminals)
                            action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                            self.actor.update(old_states, goals, action_derivs)


            elif self.hparams["buffer"][self.layer_number] == "transitions":
                if self.hparams["use_target"][self.layer_number]:
                    for _ in range(num_updates):
                        # Update weights of non-target networks
                        if self.transitions_buffer.get_current_size() > 2:
                            transitions = self.transitions_buffer.sample(self.batch_size)
                            old_states = transitions['o']
                            actions = transitions['u']
                            rewards = transitions['r']
                            new_states = transitions['o_2']
                            goals = transitions['g']
                            is_terminals = transitions['is_t']

                            self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_target_action(new_states,goals), is_terminals)
                            action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                            self.actor.update(old_states, goals, action_derivs)
                    # Update weights of target networks
                    self.sess.run(self.critic.update_target_weights)
                    self.sess.run(self.actor.update_target_weights)

                else:
                    for _ in range(num_updates):
                        # Update weights of non-target networks
                        if self.transitions_buffer.get_current_size() > 250:
                            transitions = self.transitions_buffer.sample(self.batch_size)
                            old_states = transitions['o']
                            actions = transitions['u']
                            rewards = transitions['r']
                            new_states = transitions['o_2']
                            goals = transitions['g']
                            is_terminals = transitions['is_t']


                            self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_action(new_states,goals), is_terminals)
                            action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                            self.actor.update(old_states, goals, action_derivs)

            else:
                assert False

        else:
            assert False
            


        

        
