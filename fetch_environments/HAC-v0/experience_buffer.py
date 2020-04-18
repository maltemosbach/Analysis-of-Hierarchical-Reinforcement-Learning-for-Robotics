import numpy as np
import time

class ExperienceBuffer():

    def __init__(self, max_buffer_size, batch_size):
        """Original HAC experience buffer
        Args:
            max_buffer_size (int): maximum buffer size in transitions
            batch_size (int): size of the batches that should be sampled
        """


        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert type(experience[5]) == bool

        #print("Adding transition to experience_buffer (s, a, r, s, g, t, grip_info):", experience)
        #transitions = {}
        #transitions['o'] = experience[0]
        #transitions['u'] = experience[1]
        #transitions['r'] = experience[2]
        #transitions['o_2'] = experience[3]
        #transitions['g'] = experience[4]
        #transitions['is_t'] = experience[5]

        #print("Experience buffer transitions:", transitions)


        self.experiences.append(experience)
        self.size += 1

        # If replay buffer is filled, remove a percentage of replay buffer.  Only removing a single transition slows down performance
        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size/6))
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch(self):
        states, actions, rewards, new_states, goals, is_terminals = [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=min(self.size, self.batch_size))

        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            goals.append(self.experiences[i][4])
            is_terminals.append(self.experiences[i][5])

        return states, actions, rewards, new_states, goals, is_terminals
