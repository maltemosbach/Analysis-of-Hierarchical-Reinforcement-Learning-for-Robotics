import numpy as np

class ExperienceBuffer():

    def __init__(self, max_buffer_size, batch_size, state_normalizer, goal_normalizer):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert type(experience[5]) == bool
        self.state_normalizer.update(experience[0])
        self.goal_normalizer.update(experience[4])

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
