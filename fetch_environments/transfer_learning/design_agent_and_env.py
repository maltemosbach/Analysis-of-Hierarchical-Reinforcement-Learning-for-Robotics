import numpy as np
from environment import Environment
from agent import Agent

def design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams):
    """Script that designs and creates the agent and environment.
        Args:
            FLAGS: flags determining how the alogirthm is run
            writer: writer for tensorboard logging
            writer_graph: only writes tensorflow graph to tensorboard
            sess: tensorflow session
            hparams: hyperparameters set in run.py
        """

    Writer = writer
    Writer_graph = writer_graph
    Sess = sess
    Hparams = hparams


    goal_space_test = [[1.15, 1.45], [0.5, 1.0], [0.4, 0.45]]


    # Projection functions from state to endgoal and sub-goals
    if hparams["env"] == "FetchReach-v1":
        project_state_to_end_goal = lambda sim, state: state[0:3]
        project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[0] > 1.55 else 1.05 if state[0] < 1.05 else state[0], 1.1 if state[1] > 1.1 else 0.4 if state[1] < 0.4 else state[1], 1.1 if state[2] > 1.1 else 0.4 if state[2] < 0.4 else state[2]])
    elif hparams["env"] == "FetchPush-v1" or hparams["env"] == "FetchPush_obstacle-v1" or hparams["env"] == "FetchPush_obstacle-v2" or hparams["env"] == "FetchPickAndPlace-v1" or hparams["env"] == "FetchPickAndPlace_obstacle-v1" or hparams["env"] == "FetchPickAndPlace_obstacle-v2":
        project_state_to_end_goal = lambda sim, state: state[3:6]
        project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[3] > 1.55 else 1.05 if state[3] < 1.05 else state[3], 1.1 if state[4] > 1.1 else 0.4 if state[4] < 0.4 else state[4], 1.1 if state[5] > 1.1 else 0.4 if state[5] < 0.4 else state[5]])
    else:
        assert False, "Unknown environment given."

    dist_threshold = 0.05
    end_goal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])

    # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.
    subgoal_bounds = np.array([[1.05, 1.55], [0.4, 1.1], [0.4, 1.1]])

    # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
    subgoal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])


    agent_params = {}

    # Define subgoal penalty for missing subgoal.  Please note that by default the Q value target for missed subgoals does not include Q-value of next state (i.e, discount rate = 0).  As a result, the Q-value target for missed subgoal just equals penalty.  For instance in this 3-level UR5 implementation, if a level proposes a subgoal and misses it, the Q target value for this action would be -10.  To incorporate the next state in the penalty, go to the "penalize_subgoal" method in the "layer.py" file.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale

    # Provide training schedule for agent.  Training by default will alternate between exploration and testing.  Hyperparameter below indicates number of exploration episodes.  Testing occurs for 100 episodes.  To change number of testing episodes, go to "ran_HAC.py".
    agent_params["num_exploration_episodes"] = 100



    # Instantiate and return agent and environment
    env = Environment(hparams["env"], goal_space_test, project_state_to_end_goal, end_goal_thresholds, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, FLAGS.max_actions, FLAGS.show)

    agent = Agent(FLAGS,env,agent_params, Writer, Writer_graph, Sess, hparams)

    return agent, env
