import numpy as np
from experience_buffer import ExperienceBuffer
from time import sleep
import tensorflow as tf
from utils import normalizer
import os

from ddpg import DDPG

from baselines.her.util import convert_episode_to_batch_major, store_args

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class Layer():
    def __init__(self, layer_number, FLAGS, env, sess, writer, agent_params, hparams):
        """The new layer class fo HDDPG and HTD3
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
        self.writer = writer
        self.hparams = hparams

        self.dims = {"g" : 0, "o" : 0, "u" : 0}
        self.dims["o"] = env.state_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.dims["g"] = env.end_goal_dim
        else:
            self.dims["g"] = env.subgoal_dim

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.dims["u"] = env.action_dim
        else:
            self.dims["u"] = env.subgoal_dim


        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions

        self.current_state = None
        self.goal = None

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**6

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["episodes_to_store"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = hparams["replay_k"]

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)

        # Buffer size = transitions per attempt * attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)

        # Create normalizers
        self.state_normalizer = normalizer(size=env.state_dim, eps=0.01, default_clip_range=5)
        self.goal_normalizer = normalizer(size=env.end_goal_dim, eps=0.01, default_clip_range=5)

        # self.buffer_size = 10000000
        self.batch_size = 256
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size, self.state_normalizer, self.goal_normalizer)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []


        # Configure inputs for ddpg policy
        self.hidden = 256
        self.layers = 3
        self.action_l2 = None
        if self.layer_number == 0:
            self.action_l2 = 1.0
        else:
            self.action_l2 = 0.0

        self.policy = DDPG(self.sess, self.writer, env, hparams, self.batch_size, self.replay_buffer, self.layer_number, FLAGS, self.hidden, self.layers, self.time_limit, use_replay_buffer=hparams["use_rb"], action_l2=self.action_l2)  
        

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]

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
            return self.policy.get_actions(o, g, use_target_net=self.hparams["use_target"]), "Policy", subgoal_test
            
        else:

            if np.random.random_sample() > 0.3:
                # Choose noisy action
                action = self.add_noise(self.policy.get_actions(o, g, use_target_net=self.hparams["use_target"]), env)
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
        self.replay_buffer.add(np.copy(transition))


    def append_transitions_rb(self, env, o, u, ag, g, obs, achieved_goals, acts, goals, layer_number, goal_status, highest_layer):

        o_new = env.obs['observation']
        if layer_number == highest_layer:
            ag_new = env.project_state_to_end_goal(env.sim, o_new)
        else:
            ag_new = env.project_state_to_subgoal(env.sim, o_new)

        #for i, info_dict in enumerate(info):
        #    for idx, key in enumerate(self.info_keys):
         #       info_values[idx][t, i] = info[i][key]

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        acts.append(u.copy())
        goals.append([g.copy()])
        #print("o (1):", o)
        o[...] = o_new
        #print("o (2):", o)
        ag[...] = ag_new


    # Create action replay transition by evaluating hindsight action given original goal
    def perform_action_replay_rb(self, hindsight_action, next_state, goal_status):

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
        self.replay_buffer.add(np.copy(transition))

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
            self.replay_buffer.add(np.copy(transition_b))
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

                self.replay_buffer.add(trans_copy[index])


        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []


    # Create transition penalizing subgoal if necessary.  The target Q-value when this transition is used will ignore next state as the finished boolena = True.  Change the finished boolean to False, if you would like the subgoal penalty to depend on the next state.
    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):

        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None]

        if self.FLAGS.all_trans or self.FLAGS.penalty:
            print("Level %d Penalty Trans: " % self.layer_number, transition)

        self.replay_buffer.add(np.copy(transition))



    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False

    # Determine whether layer is finished training (Changed here to keep agent training on all levels even if a (sub)goal is reached during exploration; This ensures that all episodes have the same length)
    def return_to_higher_level_new(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.
        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number and (agent.FLAGS.test or (not agent.FLAGS.test and max_lay_achieved < agent.FLAGS.layers-1)):
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False

    # Determine whether layer is finished training (Changed here to keep agent training on all levels even if a (sub)goal is reached during exploration; This ensures that all episodes have the same length)
    def return_to_higher_level_new2(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.
        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number and agent.FLAGS.test:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False


    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, env, subgoal_test = False, episode_num = None):

        # print("\nTraining Layer %d" % self.layer_number)

        # Only print goal achievement once
        printed_achieved = False

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Adding notation from rollout.generate_rollouts()
        o = np.empty((1, self.dims['o']), np.float32)   # observations
        ag = np.empty((1, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.current_state
        self.g = self.goal

        if self.layer_number == self.FLAGS.layers-1:
            ag[:] = env.project_state_to_end_goal(env.sim, self.current_state)
        else:
            ag[:] = env.project_state_to_subgoal(env.sim, self.current_state)

        self.T = self.time_limit

        # generate episodes
        obs, achieved_goals, acts, goals = [], [], [], []



        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and agent.FLAGS.show and agent.FLAGS.layers > 1:
            env.display_subgoals(agent.goal_array)
            # env.sim.data.mocap_pos[3] = env.project_state_to_end_goal(env.sim,self.current_state)
            # print("Subgoal Pos: ", env.sim.data.mocap_pos[1])

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0

        while True:

            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(agent, env, subgoal_test)
            u = action
            if u.ndim == 1:
                u = u.reshape(1, -1)

            o_new = np.empty((1, self.dims['o']))
            ag_new = np.empty((1, self.dims['g']))
            success = np.zeros(1)

            if self.FLAGS.Q_values:
                # print("\nLayer %d Action: " % self.layer_number, action)
                if layer_number == 0:
                    print("Layer %d Q-Value: " % self.layer_number, self.policy.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(action,(1,len(action)))))
                else:
                    print("Layer %d Q-Value: " % self.layer_number, self.policy.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(action,(1,len(action)))))
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
                # print("Num Actions Taken: ", agent.steps_taken)

                if agent.steps_taken >= env.max_actions:
                    print("Out of actions (Steps: %d)" % agent.steps_taken)

                agent.current_state = next_state

                # Determine whether any of the goals from any layer was achieved and, if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

            attempts_made += 1

            # Print if goal from current layer as been achieved
            if goal_status[self.layer_number] and printed_achieved == False:
                if self.layer_number < agent.FLAGS.layers - 1:
                    print("SUBGOAL ACHIEVED")
                print("\nEpisode %d, Layer %d, Attempt %d Goal Achieved" % (episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)
                if self.layer_number == agent.FLAGS.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))

                printed_achieved = True

                    #print("episode:", episode)

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


            hindsight_u = hindsight_action
            if hindsight_u.ndim == 1:
                hindsight_u = hindsight_u.reshape(1, -1)

            # Next, create hindsight transitions if not testing
            if not agent.FLAGS.test:

                self.append_transitions_rb(env, o, hindsight_u, ag, self.g, obs, achieved_goals, acts, goals, self.layer_number, goal_status, self.FLAGS.layers-1)

                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.FLAGS.layers)


                # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])


            # Print summary of transition
            if agent.FLAGS.verbose:

                print("\nEpisode %d, Level %d, Attempt %d" % (episode_num, self.layer_number,attempts_made))
                # print("Goal Array: ", agent.goal_array, "Max Lay Achieved: ", max_lay_achieved)
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)
                if self.layer_number == agent.FLAGS.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))
                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)



            # Update state of current layer
            self.current_state = agent.current_state

            # Return to previous level to receive next subgoal if applicable
            # if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
            # if (a layer has achieved its goal and it is higher or equal to the current layer) or the agent is out of steps or the layer is out of steps:
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:

                if self.layer_number > 0:
                    print("in if fo layer", self.layer_number)
                    print("attempts_made:", attempts_made)
                    print("self.time_limit:", self.time_limit)
                    print("agent.steps_taken:", agent.steps_taken)

                if self.layer_number == agent.FLAGS.layers-1:
                    pass
                    #print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)


                if (attempts_made >= self.time_limit) and not agent.FLAGS.test or (agent.steps_taken>=env.max_actions and self.layer_number>0 and not agent.FLAGS.test):
                    if attempts_made >= self.time_limit:
                        obs.append(o.copy())
                        achieved_goals.append(ag.copy())
                    episode = dict(o=obs,
                           u=acts,
                           g=goals,
                           ag=achieved_goals)
                    if self.layer_number>0:
                        print("episode_org:", episode)
                    episode = convert_episode_to_batch_major(episode)
                    assert len(episode['o'][0]) == self.T+1, "ERROR: Episode size [o] wrong"
                    assert len(episode['u'][0]) == self.T, "ERROR: Episode size [u] wrong"
                    assert len(episode['g'][0]) == self.T, "ERROR: Episode size [g] wrong"
                    assert len(episode['ag'][0]) == self.T+1, "ERROR: Episode size [ag] wrong"
                    self.policy.store_episode(episode)
                    print("Episode stored for layer", self.layer_number)

                    

                # If not testing, finish goal replay by filling in missing goal and reward values before returning to prior level.
                if not agent.FLAGS.test:
                    if self.layer_number == agent.FLAGS.layers - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.subgoal_thresholds

                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level_new2(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, max_lay_achieved



    # Update actor and critic networks
    def learn(self, num_updates):

        if self.layer_number == 0:
            print("learn layer 0 is called!")
            # Update nets if replay buffer is large enough
            if self.hparams["use_rb"]:
                print("self.policy.buffer.get_current_size():", self.policy.buffer.get_current_size())
                if self.policy.buffer.get_current_size() >= 50:
                    # Update main nets num_updates times
                    for _ in range(num_updates):
                        self.policy.train()

                    # Update all target nets
                    self.policy.update_target_net()
            else:
                if self.replay_buffer.size >= self.batch_size:
                    # Update main nets num_updates times
                    for _ in range(num_updates):
                        self.policy.train()

                    # Update all target nets
                    self.policy.update_target_net()

        else:
            # Right now only layer 0 can learn with both buffers so layers above always use the Experience_Buffer
            #print("learn layer 1 is called!")
            if self.replay_buffer.size >= self.batch_size:
                # Update main nets num_updates times
                for _ in range(num_updates):
                    self.policy.train()

                # Update all target nets
                self.policy.update_target_net()
            