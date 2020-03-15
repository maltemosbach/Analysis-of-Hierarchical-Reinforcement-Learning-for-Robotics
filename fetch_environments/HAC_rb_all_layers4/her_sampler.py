import numpy as np
import time

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, FLAGS):
    """Creates a sample function that can be used for HER experience replay.
    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}

        buffer_size refers to the buffers size in full episodes. The first shape index can be used to choose an episode
        """

        # Episode batch is given from the replay buffer and contains all available transitions
        
        if 'is_sgtt' in episode_batch.keys():
        #    print("In her_sampler for layer 1")
            layer_number = 1
            #print("episode_batch:", episode_batch)
        else:
        #    print("In her_sampler for layer 0")
            layer_number = 0

        #print("episode_batch:", episode_batch)

        #print("episode_batch.keys():", episode_batch.keys())
        #print("episode_batch['o'].shape[1]:", episode_batch['o'].shape[1])
        #print("episode_batch['ag'].shape[1]:", episode_batch['ag'].shape[1])
        #print("episode_batch['g'].shape[1]:", episode_batch['g'].shape[1])
        #print("episode_batch['u'].shape[1]:", episode_batch['u'].shape[1])
        #print("episode_batch['o_2'].shape[1]:", episode_batch['o_2'].shape[1])
        #print("episode_batch['ag_2'].shape[1]:", episode_batch['ag_2'].shape[1])

        #print("episode_batch['o'].shape[0]:", episode_batch['o'].shape[0])
        #print("episode_batch['ag'].shape[0]:", episode_batch['ag'].shape[0])
        #print("episode_batch['g'].shape[0]:", episode_batch['g'].shape[0])
        #print("episode_batch['u'].shape[0]:", episode_batch['u'].shape[0])
        #print("episode_batch['o_2'].shape[0]:", episode_batch['o_2'].shape[0])
        #print("episode_batch['ag_2'].shape[0]:", episode_batch['ag_2'].shape[0])
        #time.sleep(3)



        
        T = episode_batch['u'].shape[1]   # Length of one episode
        rollout_batch_size = episode_batch['u'].shape[0]   # Total numer of episodes in the buffer
        batch_size = batch_size_in_transitions   # Total number of transitions to be sampled

        # Select which episodes and time steps to use.


        # Sampling which episodes to use (could maybe be relevant for improvement)
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        #print("episode_idxs:", episode_idxs)
        # Sampling which timesteps inside the episodes to use
        t_samples = np.random.randint(T, size=batch_size)
        #print("t_samples:", t_samples)
        #print("len(t_samples):", len(t_samples))

        # Sampling 256 (batch_size) amount of actual transitions form replay buffer
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.

        # Sampling about future_p amount of the total transitions which will be turned into HER transitions
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        #print("her_indexes:", her_indexes)
        #print("len(her_indexes)[0]):", len(her_indexes[0]))
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        #print("future_t:", future_t)
        #print("len(future_t):", len(future_t))
        #time.sleep(3)

        # info_sgtt is -1 if sgtt and sg has not been reached; 1 if sgtt and sg has been reached; 0 if not sgtt
        if layer_number == 1:
            #print("transitions:", transitions)
            #print("transitions['is_sgtt']:", transitions['is_sgtt'])
            sg_testing_idxs = np.where(transitions['is_sgtt'] != [0])[0]
            #print("sg_testing_idxs:", sg_testing_idxs)


            # From all those transitions sampled getting the ones which are subgoal testing transitions
            sg_testing_trans = {key: transitions[key][sg_testing_idxs] for key in transitions.keys()}
            #print("sg_testing_trans:", sg_testing_trans)

            penalizing_idxs = np.where(transitions['is_sgtt'] == [-1])[0]
            penalizing_transitions = {key: transitions[key][penalizing_idxs] for key in transitions.keys()}
            penalizing_transitions['r'] = -FLAGS.time_scale * np.ones(penalizing_transitions['is_sgtt'].shape[0])
            #penalizing_transitions['r'] = penalizing_transitions['r'].reshape(penalizing_transitions['is_sgtt'].shape)
            #print("penalizing_transitions:", penalizing_transitions)



        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag


        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        transitions['r'] = reward_fun(**reward_params)

        #print("transitions2:", transitions)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        #print("transitions:", transitions)
        if layer_number == 1:
            all_transitions = {}
            all_transitions = {key: np.concatenate([transitions[key], penalizing_transitions[key]]) for key in transitions.keys()}
            #all_transitions['r'] = np.concatenate([transitions['r'], penalizing_transitions['r']])
            #all_transitions['u'] = np.concatenate([transitions['u'], penalizing_transitions['u']])
            #print("all_transitions:", all_transitions)
            return all_transitions
        #time.sleep(1000)

        return transitions

    return _sample_her_transitions