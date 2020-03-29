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

        print("self.finished_transitions.items()", self.finished_transitions.items())


        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.size = 10

        self.replay_k = replay_k





    def store_episode(self, episode_batch):
        """episode_batch: array(timesteps x dim_key)
        """
        # Batch includes o, ag, g, u
        episode_batch = {key: episode_batch[key][0] for key in episode_batch.keys()}

        batch_sizes = {key: episode_batch[key].shape for key in episode_batch.keys()}

        print("batch_sizes:", batch_sizes)

        episode_batch['o_2'] = episode_batch['o'][1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][1:, :]

        # raw episode batch finished with o, ag, g, u, o_2, ag_2
        # would get passed to her_sampler now
        print("episode_batch:", episode_batch)

        # length (timesteps) of this episode
        T = batch_sizes['u'][0]
        print("T:", T)

        # Calculate number of total transitions to store
        total_num_new_trans = (1+self.replay_k) * T

        #print("total_num_new_trans:", total_num_new_trans)

        #print("np.repeat(episode_batch['u'], 3, axis=0):", np.repeat(episode_batch['u'], 3, axis=0))
        print("np.where(np.arange((1 + self.replay_k) * T) % (1+ self.replay_k) != 0) :", np.where(np.arange((1 + self.replay_k) * T) % (1+ self.replay_k) != 0))

        transitions = {key: np.repeat(episode_batch[key], 1 + self.replay_k, axis=0) for key in episode_batch.keys()}

        indexes = np.arange((1 + self.replay_k) * T)

        timesteps = np.repeat(np.arange(T), 1 + self.replay_k)

        her_indexes = np.where(np.arange((1 + self.replay_k) * T) % (1+ self.replay_k) != 0)

        print("indexes:", indexes)
        print("her_indexes:", her_indexes)
        print("timesteps:", timesteps)

        future_offset = np.random.uniform(size=total_num_new_trans) * (T - timesteps)
        future_offset = future_offset.astype(int)
        print("future_offset:", future_offset)

        future_t = (timesteps + 1 + future_offset)[her_indexes]

        print("future_t:", future_t)

        future_ag = episode_batch['ag'][future_t]

        print("future_ag:", future_ag)

        transitions['g'][her_indexes] = future_ag

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        transitions['r'] = self.reward_fun(**reward_params)



        print("transitions['r']:", transitions['r'])


        time.sleep(1000)

        



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