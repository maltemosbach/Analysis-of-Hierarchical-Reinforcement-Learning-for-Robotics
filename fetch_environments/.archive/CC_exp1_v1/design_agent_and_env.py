"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
from environment import Environment
from utils import check_validity
from agent import Agent

def design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams):

    Writer = writer
    Writer_graph = writer_graph
    Sess = sess
    Hparams = hparams

    """
    1. DESIGN AGENT

    The key hyperparameters for agent construction are

        a. Number of levels in agent hierarchy
        b. Max sequence length in which each policy will specialize
        c. Max number of atomic actions allowed in an episode
        d. Environment timesteps per atomic action

    See Section 3 of this file for other agent hyperparameters that can be configured.
    """

    FLAGS.layers = hparams["layers"]    # Enter number of levels in agent hierarchy

    FLAGS.time_scale = 10    # Enter max sequence length in which each policy will specialize

    # Enter max number of atomic actions.  This will typically be FLAGS.time_scale**(FLAGS.layers).  However, in the UR5 Reacher task, we use a shorter episode length.
    max_actions = 50

    timesteps_per_action = 1    # Provide the number of time steps per atomic action.


    """
    2. DESIGN ENVIRONMENT

        a. Designer must provide the original UMDP (S,A,T,G,R).
            - The S,A,T components can be fulfilled by providing the Mujoco model.
            - The user must separately specifiy the initial state space.
            - G can be provided by specifying the end goal space.
            - R, which by default uses a shortest path {-1,0} reward function, can be implemented by specifying two components: (i) a function that maps the state space to the end goal space and (ii) the end goal achievement thresholds for each dimensions of the end goal.

        b.  In order to convert the original UMDP into a hierarchy of k UMDPs, the designer must also provide
            - The subgoal action space, A_i, for all higher-level UMDPs i > 0
            - R_i for levels 0 <= i < k-1 (i.e., all levels that try to achieve goals in the subgoal space).  As in the original UMDP, R_i can be implemented by providing two components:(i) a function that maps the state space to the subgoal space and (ii) the subgoal achievement thresholds.

        c.  Designer should also provide subgoal and end goal visualization functions in order to show video of training.  These can be updated in "display_subgoal" and "display_end_goal" methods in the "environment.py" file.

    """

    # Provide file name of Mujoco model(i.e., "pendulum.xml").  Make sure file is stored in "mujoco_files" folder
    model_name = "FetchReach.xml"

    initial_state_space = [[1.05, 1.55], [0.4, 1.1], [0.4, 0.7], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

    # Provide end goal space.  The code supports two types of end goal spaces if user would like to train on a larger end goal space.  If user needs to make additional customizations to the end goals, the "get_next_goal" method in "environment.py" can be updated.
    #goal_space_train = [[1.05, 1.55], [0.4, 1.1], [0.4, 0.9]]
    goal_space_train = [[1.15, 1.45], [0.5, 1.0], [0.4, 0.45]]
    goal_space_test = goal_space_train


    # Provide a function that maps from the state space to the end goal space.  This is used to (i) determine whether the agent should be given the sparse reward and (ii) for Hindsight Experience Replay to determine which end goal was achieved after a sequence of actions.
    project_state_to_end_goal = lambda sim, state: state[3:6]

    # For the FetchReach task the end goal is the desired position of the gripper
    dist_threshold = 0.05
    end_goal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])


    # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.

    subgoal_bounds = np.array([[1.05, 1.55], [0.4, 1.1], [0.4, 0.7]])


    # Provide state to subgoal projection function.
    project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[3] > 1.55 else 1.05 if state[3] < 1.05 else state[3], 1.1 if state[4] > 1.1 else 0.4 if state[4] < 0.4 else state[4], 0.7 if state[5] > 0.7 else 0.4 if state[5] < 0.4 else state[5]])

    # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
    subgoal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])


    # To properly visualize goals, update "display_end_goal" and "display_subgoals" methods in "environment.py"


    """
    3. SET MISCELLANEOUS HYPERPARAMETERS

    Below are some other agent hyperparameters that can affect results, including
        a. Subgoal testing percentage
        b. Subgoal penalty
        c. Exploration noise
        d. Replay buffer size
    """

    agent_params = {}

    # Define percentage of actions that a subgoal level (i.e. level i > 0) will test subgoal actions
    agent_params["subgoal_test_perc"] = hparams["sg_test_perc"]

    # Define subgoal penalty for missing subgoal.  Please note that by default the Q value target for missed subgoals does not include Q-value of next state (i.e, discount rate = 0).  As a result, the Q-value target for missed subgoal just equals penalty.  For instance in this 3-level UR5 implementation, if a level proposes a subgoal and misses it, the Q target value for this action would be -10.  To incorporate the next state in the penalty, go to the "penalize_subgoal" method in the "layer.py" file.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale


    # Define number of episodes of transitions to be stored by each level of the hierarchy
    # To store up to 1.000.000 transitions this should be >= 20.000
    agent_params["episodes_to_store"] = 20000

    # Provide training schedule for agent.  Training by default will alternate between exploration and testing.  Hyperparameter below indicates number of exploration episodes.  Testing occurs for 100 episodes.  To change number of testing episodes, go to "ran_HAC.py".
    agent_params["num_exploration_episodes"] = 100

    # For other relavent agent hyperparameters, please refer to the "agent.py" and "layer.py" files



    # Instantiate and return agent and environment
    env = Environment(model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions, timesteps_per_action, FLAGS.show)

    agent = Agent(FLAGS,env,agent_params, Writer, Writer_graph, Sess, hparams)

    return agent, env
