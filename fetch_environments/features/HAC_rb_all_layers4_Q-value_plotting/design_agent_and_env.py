"""
This file provides the template for designing the agent and environment. When running on th same environment all relevant hyperparameters can be altered in initialize_HAC.py
"""

import numpy as np
from environment import Environment
from agent import Agent

def design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams):

    Writer = writer
    Writer_graph = writer_graph
    Sess = sess
    Hparams = hparams

    FLAGS.layers = hparams["layers"]    # Enter number of levels in agent hierarchy

    FLAGS.time_scale = 10    # Enter max sequence length in which each policy will specialize

    # Enter max number of atomic actions.  This will typically be FLAGS.time_scale**(FLAGS.layers).  However, in the UR5 Reacher task, we use a shorter episode length.
    max_actions = 50

    goal_space_test = [[1.15, 1.45], [0.5, 1.0], [0.4, 0.45]]

    # Provide a function that maps from the state space to the end goal space.  This is used to (i) determine whether the agent should be given the sparse reward and (ii) for Hindsight Experience Replay to determine which end goal was achieved after a sequence of actions.
    # For the FetchReach task the end goal is the desired position of the gripper, for fetchPush and fetchPickAndPlace it is the desired position of the object
    project_state_to_end_goal = lambda state: state[0:3]
    
    dist_threshold = 0.05
    end_goal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])


    # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.
    subgoal_bounds = np.array([[1.05, 1.55], [0.4, 1.1], [0.4, 0.7]])

    # Provide state to subgoal projection function.
    project_state_to_subgoal = lambda state: np.array([1.55 if state[0] > 1.55 else 1.05 if state[0] < 1.05 else state[0], 1.1 if state[1] > 1.1 else 0.4 if state[1] < 0.4 else state[1], 0.7 if state[2] > 0.7 else 0.4 if state[2] < 0.4 else state[2]])

    subgoal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])


    agent_params = {}

    # Define percentage of actions that a subgoal level (i.e. level i > 0) will test subgoal actions
    agent_params["subgoal_test_perc"] = hparams["sg_test_perc"]

    # Define subgoal penalty for missing subgoal.  Please note that by default the Q value target for missed subgoals does not include Q-value of next state (i.e, discount rate = 0).  As a result, the Q-value target for missed subgoal just equals penalty.  For instance in this 3-level UR5 implementation, if a level proposes a subgoal and misses it, the Q target value for this action would be -10.  To incorporate the next state in the penalty, go to the "penalize_subgoal" method in the "layer.py" file.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale

    # Define exploration noise that is added to both subgoal actions and atomic actions.  Noise added is Gaussian N(0, noise_percentage * action_dim_range)
    agent_params["atomic_noise"] = [hparams["ac_n"] for i in range(4)]
    agent_params["subgoal_noise"] = [hparams["sg_n"] for i in range(len(subgoal_thresholds))]

    # Define number of episodes of transitions to be stored by each level of the hierarchy3
    agent_params["episodes_to_store"] = 20000

    # Provide training schedule for agent.  Training by default will alternate between exploration and testing.  Hyperparameter below indicates number of exploration episodes. 
    agent_params["num_exploration_episodes"] = 2

    # Instantiate and return agent and environment
    env = Environment(goal_space_test, project_state_to_end_goal, end_goal_thresholds, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions, FLAGS.show)

    agent = Agent(FLAGS,env,agent_params, Writer, Writer_graph, Sess, hparams)

    return agent, env
