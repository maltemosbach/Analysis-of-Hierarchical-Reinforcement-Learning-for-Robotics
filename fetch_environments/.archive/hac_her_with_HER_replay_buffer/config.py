from baselines.her.her_sampler import make_sample_her_transitions
import os
import numpy as np
import gym



def configure_her(env):

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
        'replay_strategy' : 'future',
        'replay_k' : 4
    }
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions