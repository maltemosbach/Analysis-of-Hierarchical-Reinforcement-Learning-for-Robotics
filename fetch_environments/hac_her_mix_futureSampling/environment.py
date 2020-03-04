from tkinter import *
from tkinter import ttk
import time
import numpy as np
import gym

class Environment():

    def __init__(self, model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions = 1200, num_frames_skip = 10, show = False):

        self.name = model_name

        # Create Gym simulation
        self.gymEnv = gym.make('FetchReach-v1')

        #self.sim = self.gymEnv.env.sim
        self.sim = "Test"

        '''
        print("- - - - - - - - - - INTRODUCTION - - - - - - - - - -")
        print("self.gymEnv.observation_space:", self.gymEnv.observation_space)
        print("self.gymEnv.observation_space[observation]:", self.gymEnv.observation_space["observation"])
        print("self.gymEnv.observation_space[observation].shape[0]:", self.gymEnv.observation_space["observation"].shape[0])
        print("self.gymEnv.action_space:", self.gymEnv.action_space)
        print("self.gymEnv.action_space.shape[0]:", self.gymEnv.action_space.shape[0])
        print("self.gymEnv.action_space.high:", self.gymEnv.action_space.high)
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - ")
        '''

        self.state_dim = self.gymEnv.observation_space["observation"].shape[0] # State will include [gripper_pos_x, gripper_pos_y, gripper_pos_z, grip_opening, grip_angle, gripper_velo_x, gripper_velo_y, gripper_velo_z, grip_velo_opening, grip_velo_angle]
        self.action_dim = self.gymEnv.action_space.shape[0] # low-level actions will include [delta_x, delta_y, delta_z, delta_grip]
        self.action_bounds = self.gymEnv.action_space.high # low-level action bounds as given by the gym environment
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal


        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean

        self.numTimesteps = 0

        '''
        print("- - - - - - - - - - V A L I D I T Y - - - - - - - - - -")
        print("state_dim:", self.state_dim)
        print("action_dim:", self.action_dim)
        print("action_bounds:", self.action_bounds)
        print("action_offset:", self.action_offset)
        print("end_goal_dim:", self.end_goal_dim)
        print("subgoal_dim:", self.subgoal_dim)
        print("subgoal_bounds", self.subgoal_bounds)
        print("end_goal_thresholds:", self.end_goal_thresholds)
        print("subgoal_thresholds:", self.subgoal_thresholds)
        print("initial_state_space:", self.initial_state_space)
        print("goal_space_test:", self.goal_space_test)
        print("max_actions:", self.max_actions)
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        '''



    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return self.obs["observation"]

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal = None):
        self.obs = self.gymEnv.reset()
        self.done = False
        self.numTimesteps = 0

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):
        self.obs, self.rewards, self.done, self.info = self.gymEnv.step(action)
        self.numTimesteps += 1
        #print("Timestep:", self.numTimesteps)
        #if self.done:
            #print("Done after", self.numTimesteps, "timesteps")
        if self.visualize:
            self.gymEnv.render()

        return self.get_state()


    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self,end_goal):
        # Goal can be visualized by changing the location of the relevant site object.
        sites_offset = (self.gymEnv.env.sim.data.site_xpos - self.gymEnv.env.sim.model.site_pos).copy()
        site_id = self.gymEnv.env.sim.model.site_name2id('endgoal')
        #print("sites_offset:", sites_offset)
        self.gymEnv.env.sim.model.site_pos[site_id] = end_goal - sites_offset[1]
        self.gymEnv.env.sim.model.site_rgba[1][3] = 0.5

        # Don't show gym endgoal
        site_id = self.gymEnv.env.sim.model.site_name2id('target0')
        self.gymEnv.env.sim.model.site_pos[site_id] = end_goal - sites_offset[0]
        self.gymEnv.env.sim.model.site_rgba[0][3] = 0.1


    # Function returns an end goal
    def get_next_goal(self,test):
        end_goal = np.zeros((len(self.goal_space_test)))

        useGymGoals = True

        if useGymGoals:
            end_goal = self.obs["desired_goal"]
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])

        # Visualize End Goal
        self.display_end_goal(end_goal)
        return end_goal


    # Visualize all subgoals
    def display_subgoals(self,subgoals):
        # subgoals array contains [subgoal1, subgoal2, endgoal]
        #print("self.subgoal_bounds:", self.subgoal_bounds)
        print("subgoals:", subgoals)

        if len(subgoals) == 2:
            sites_offset = (self.gymEnv.env.sim.data.site_xpos - self.gymEnv.env.sim.model.site_pos).copy()
            site_id = self.gymEnv.env.sim.model.site_name2id('subgoal1')
            self.gymEnv.env.sim.model.site_pos[site_id] = subgoals[0][:3] - sites_offset[2]
            self.gymEnv.env.sim.model.site_rgba[2][3] = 0.5


        if len(subgoals) == 3:
            sites_offset = (self.gymEnv.env.sim.data.site_xpos - self.gymEnv.env.sim.model.site_pos).copy()
            site_id = self.gymEnv.env.sim.model.site_name2id('subgoal1')
            self.gymEnv.env.sim.model.site_pos[site_id] = subgoals[0][:3] - sites_offset[2]
            site_id = self.gymEnv.env.sim.model.site_name2id('subgoal2')
            self.gymEnv.env.sim.model.site_pos[site_id] = subgoals[1][:3] - sites_offset[3]
            self.gymEnv.env.sim.model.site_rgba[2][3] = 0.5
            self.gymEnv.env.sim.model.site_rgba[3][3] = 0.5
