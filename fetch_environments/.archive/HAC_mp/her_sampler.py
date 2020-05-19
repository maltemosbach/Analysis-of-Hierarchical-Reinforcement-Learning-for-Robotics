import numpy as np
import time

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, FLAGS):
    """Creates a sample function that can be used for HER experience replay inside HAC algorithm
    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
        FLAGS: used for penalty in subgoal testing transitions
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: list(buffer_size x arrays(T x dim_key))}

        buffer_size refers to the buffers size in full episodes. The first shape index can be used to choose an episode
        """

        # Episode batch is given from the replay buffer and contains all available transitions
        
        if 'is_sgtt' in episode_batch.keys():
            layer_number = 1
        else:
            layer_number = 0


        # Total numer of episodes in the buffer
        rollout_batch_size = len(episode_batch['u'])

        # Length of each episode
        T = np.array([episode_batch['u'][i].shape[0] for i in range(rollout_batch_size)], dtype=int)


        batch_size = batch_size_in_transitions   # Total number of transitions to be sampled

        # Select which episodes and time steps to use.
        # Sampling which episodes to use (could maybe be relevant for improvement)
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        # Sampling which timesteps inside the episodes to use
        t_samples = np.zeros(batch_size, dtype=int)
        check = np.empty(batch_size)

        for i in range(batch_size):
            t_samples[i] = np.random.randint(T[episode_idxs[i]])

        episode_idxs.reshape(1, -1)


        # Sampling 256 (batch_size) amount of actual transitions form replay buffer
        transitions = {key: [] for key in episode_batch.keys()}

        for i in range(batch_size):
            for key in episode_batch.keys():
                transitions[key].append(episode_batch[key][episode_idxs[i]][t_samples[i]])


        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.

        # Sampling about future_p * batch_size amount of the total transitions which will be turned into HER transitions
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        her_indexes = her_indexes[0]
        future_offset = np.random.uniform(size=batch_size) * (T[episode_idxs] - t_samples)
        future_offset = future_offset.astype(int)
        
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # info_sgtt is -1 if sgtt and sg has not been reached; 1 if sgtt and sg has been reached; 0 if not sgtt
        if layer_number == 1:

            penalizing_idxs = np.where(transitions['is_sgtt'] == [-1])[0]
            penalizing_transitions = {key: transitions[key][penalizing_idxs] for key in transitions.keys()}
            penalizing_transitions['r'] = -FLAGS.time_scale * np.ones(penalizing_transitions['is_sgtt'].shape[0])


        # Replace goal with a future ag form the episode but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = []
    
        for i in range(her_indexes.shape[0]):
            j = episode_idxs[her_indexes][i]
            k = future_t[i]
            future_ag.append(episode_batch['ag'][j][k])
            transitions['g'][her_indexes[i]] = future_ag[i]


        # Convert back from "lists-style" replay buffer to arrays
        for key in transitions.keys():
            transitions[key] = np.asarray(transitions[key])


        # Calculate reward for all transitions
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        # Check total amount of sampled transitions
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        # Add penalty transitions from subgoal testing to transitions batch
        if layer_number > 0:
            all_transitions = {}
            all_transitions = {key: np.concatenate([transitions[key], penalizing_transitions[key]]) for key in transitions.keys()}
            return all_transitions


        return transitions

    return _sample_her_transitions