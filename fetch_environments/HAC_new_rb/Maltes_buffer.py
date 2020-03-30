import numpy as np
import time


class MaltesBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, replay_k, reward_fun):
        """Creates my new buffer.
        Args:
            buffer_shapes (dict of ints): the shape of all arguments of the buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size_in_transitions = size_in_transitions
        self.reward_fun = reward_fun



        # Should include o, g, u, r, o_2, is_t
        # {key: array(transition_number, key_shape)}
        self.finished_transitions = {key: np.empty([self.size_in_transitions, buffer_shapes[key]])
                                    for key in buffer_shapes.keys()}

        self.finished_transitions['is_t'] = np.empty([self.size_in_transitions, 1], dtype=bool)

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.size = size_in_transitions

        assert replay_k >= 0
        self.replay_k = replay_k





    def store_episode(self, episode_batch):
        """episode_batch: array(timesteps x dim_key)
        """
        # Batch includes o, ag, g, u
        episode_batch = {key: episode_batch[key][0] for key in episode_batch.keys()}

        batch_sizes = {key: episode_batch[key].shape for key in episode_batch.keys()}

        #print("batch_sizes:", batch_sizes)
        # length (timesteps) of this episode
        T = batch_sizes['u'][0]
        #print("T:", T)


        # Creating new observation and new achieved goal
        episode_batch['o_2'] = episode_batch['o'][1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][1:, :]

        # Removing last o_2 from inital observations and same for ag
        episode_batch['o'] = episode_batch['o'][:-1, :]
        episode_batch_save = {}
        episode_batch_save['ag'] = episode_batch['ag']
        episode_batch['ag'] = episode_batch['ag'][:-1, :]

        for key in episode_batch.keys():
            #print("episode_batch[{k}].shape[0]=".format(k=key), episode_batch[key].shape[0])
            assert episode_batch[key].shape[0] == T

        # raw episode batch finished with o, ag, g, u, o_2, ag_2
        # would get passed to her_sampler now
        #print("episode_batch:", episode_batch)

        
        # Calculate number of total transitions to store
        total_num_new_trans = (1+self.replay_k) * T

        #print("total_num_new_trans:", total_num_new_trans)

        #print("np.repeat(episode_batch['u'], 3, axis=0):", np.repeat(episode_batch['u'], 3, axis=0))
        #print("np.where(np.arange((1 + self.replay_k) * T) % (1+ self.replay_k) != 0) :", np.where(np.arange((1 + self.replay_k) * T) % (1+ self.replay_k) != 0))

        transitions = {key: np.repeat(episode_batch[key], 1 + self.replay_k, axis=0) for key in episode_batch.keys()}

        indexes = np.arange((1 + self.replay_k) * T)

        timesteps = np.repeat(np.arange(T), 1 + self.replay_k)

        her_indexes = np.where(np.arange((1 + self.replay_k) * T) % (1+ self.replay_k) != 0)

        #print("indexes:", indexes)
        #print("her_indexes:", her_indexes)
        #print("timesteps:", timesteps)

        future_offset = np.random.uniform(size=total_num_new_trans) * (T - timesteps)
        future_offset = future_offset.astype(int)
        #print("future_offset:", future_offset)

        future_t = (timesteps + 1 + future_offset)[her_indexes]

        #print("future_t:", future_t)

        future_ag = episode_batch_save['ag'][future_t]

        #print("future_ag:", future_ag)


        # Substituting goal with a future achieved state from the same episode for all her transitions
        transitions['g'][her_indexes] = future_ag



        # Calculating reward for all transitions
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        transitions['r'] = self.reward_fun(**reward_params)

        # Adding 'is terminal' information to all transitions
        transitions['is_t'] = np.empty(total_num_new_trans, dtype=bool)
        for l in range(total_num_new_trans):
            transitions['is_t'][l] = bool(transitions['r'][l] == 0)

        transitions['r'] = transitions['r'].reshape(total_num_new_trans, 1)
        transitions['is_t'] = transitions['is_t'].reshape(total_num_new_trans, 1)


        #print("self.finished_transitions['u'][idxs].shape:", self.finished_transitions['u'][idxs].shape)
        #print("transitions['u'].shape:", transitions['u'].shape)
        #print("self.finished_transitions['o'][idxs].shape:", self.finished_transitions['o'][idxs].shape)
        #print("transitions['o'].shape:", transitions['o'].shape)
        #print("self.finished_transitions['r'][idxs].shape:", self.finished_transitions['r'][idxs].shape)
        #print("transitions['r'].shape:", transitions['r'].shape)
        #print("self.finished_transitions['is_t'][idxs].shape:", self.finished_transitions['is_t'][idxs].shape)
        #print("transitions['is_t'].shape:", transitions['is_t'].shape)



        # Get indexes where to store transitions
        idxs = self._get_storage_idx(total_num_new_trans)
        #print("idxs:", idxs)


        # load inputs into buffers
        for key in self.finished_transitions.keys():
            self.finished_transitions[key][idxs] = transitions[key]

        



    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx

    def get_current_size(self):
        return self.current_size



    def clear_buffer(self):
        self.current_size = 0



    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """

        #print("sample new buffer is called!")

        transitions = {}
        t_samples = np.random.randint(self.current_size, size=batch_size)
        #print("t_samples:", t_samples)

        assert self.current_size > 0
        for key in self.finished_transitions.keys():
            transitions[key] = self.finished_transitions[key][t_samples]

        transitions['r'] = transitions['r'].reshape(batch_size,)


        #print("transitions:", transitions)

        return transitions















