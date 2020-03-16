"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

import pickle as cpickle
import agent as Agent
from utils import print_summary
from tensorboardX import SummaryWriter
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

import numpy as np


NUM_BATCH = 7
TEST_FREQ = 2

num_test_episodes = 10

success_rate_plt = np.zeros(np.ceil(NUM_BATCH/2).astype(int))
x_axis = np.arange(0.0, np.ceil(NUM_BATCH/2), 1.0)

def run_HAC(FLAGS,env,agent,writer,sess):

    Writer = writer
    Sess = sess

    # Print task summary
    print_summary(FLAGS,env)
    i = 0
    
    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True
     
    for batch in range(NUM_BATCH):

        num_episodes = agent.other_params["num_exploration_episodes"]
        
        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            num_episodes = num_test_episodes            

            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(num_episodes):
            
            print("\nBatch %d, Episode %d" % (batch, episode))
            
            # Train for an episode
            success = agent.train(env, episode)

            if success:
                print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                
                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_episodes += 1            

        # Save agent
        agent.save_model(episode)
        agent.save_lowest_layer(episode)
           
        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:
            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            success_rate_plt[i] = success_rate/100
            Writer.add_scalar("success_rate", success_rate/100, i)
            agent.log_tb(i)
            i += 1
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")




    if FLAGS.play:
        input("Play the trained agent ...")
        agent.FLAGS.show = True
        agent.FLAGS.test = True
        env.visualize = True
        try:
            while True:
                episode = 0
                print("\nEpisode %d" % (episode))
            
                # Test for an episode
                success = agent.train(env, episode)
                episode += 1

                if success:
                    print("Episode %d End Goal Achieved\n" % (episode))

        except KeyboardInterrupt:
            pass


    # For Q-value visualozation (WIP)
    if agent.FLAGS.layers > 0:
        print("Q-val is called!")
        o = np.empty([10, 14, 10])
        g = np.array([1.05, 0.4, 0.5])
        u = np.empty((10, 14, 4)) # grid dim, grid dim, action dim
        Q_vals = np.empty((10, 14))
        Q_vals2 = np.empty((2))

        u[0, 0, :] = np.array([1.05, 0.4, 0.5, 0])

        for i in range(10):
            for j in range(14):
                u[i, j, :] = np.random.rand(4)
                o[i, j, :] = np.array([1.075 + i*0.05, 0.425 +j*0.05, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                Q_vals[i, j] = agent.layers[0].policy.get_Q_values(o[i, j, :], g, np.array([0, 0, 0, 0]), use_target_net=False)


        Q_vals2[0] = agent.layers[0].policy.get_Q_values_new(np.array([1.5, 1.0, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), g, np.array([-1, -1, 0, 0]), use_target_net=False)
        Q_vals2[1] = agent.layers[0].policy.get_Q_values_new(np.array([1.5, 1.0, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), g, np.array([1, 1, 0, 0]), use_target_net=False)
        print("o:", o)
        print("u:", u)
        print("Q_vals:", Q_vals)
        print("Q_vals2:", Q_vals2)


    return np.copy(success_rate_plt), np.copy(Q_vals)


    

     
