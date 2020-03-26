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



def run_HAC(FLAGS,env,agent,writer,sess, NUM_BATCH):

    TEST_FREQ = 2
    num_test_episodes = 10

    success_rate_plt = np.zeros(np.ceil(NUM_BATCH/2).astype(int))
    critic_loss_layer0 = -1*np.ones(np.ceil(NUM_BATCH/2).astype(int))
    critic_loss_layer1 = -1*np.ones(np.ceil(NUM_BATCH/2).astype(int))
    # Print task summary
    print_summary(FLAGS,env)
    ind = 0


    # Create Q_val_tables (step (0,1), layer (0,1), x-dim (10), y-dim (14))
    Q_val_table = np.ones((2, 2, 10, 14))

    
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
            success_rate_plt[ind] = success_rate/100
            writer.add_scalar("success_rate", success_rate/100, ind)
            Critic_losses = agent.log_tb(ind)
            critic_loss_layer0[ind] = Critic_losses[0]
            if agent.hparams["layers"] > 1:
                critic_loss_layer1[ind] = Critic_losses[1]

            ind += 1
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")

        # Create Q-function matrix if it is the first or last batch
        if batch == 5 or batch == NUM_BATCH-1:
            Q_vals_layer_0 = np.ones((10, 14))
            Q_vals_layer_1 = np.ones((10, 14))

            # Defining observations for different fetch envs
            if env.name == "FetchReach-v1":
                o = np.zeros([10, 14, 10])
            elif env.name == "FetchPush-v1" or env.name == "FetchPickAndPlace-v1":
                o = np.zeros([10, 14, 25])

            # - - - - Q-vals for FetchReach - - - - 
            # Goal is placed near the top left of the plane. For layer 0 the possible states in the plane are evaluated.
            # For layer 1 the position of the gripper is the closer to the bottom right and the actions (subgoals) in the plane
            # are evaluated.
            if env.name == "FetchReach-v1":
                g = np.array([1.15, 0.6, 0.5])
                Q_vals_layer_0 = np.ones((10, 14))

                for i in range(10):
                    for j in range(14):
                        o[i, j, :] = np.array([1.075 + i*0.05, 0.425 +j*0.05, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        Q_vals_layer_0[i, j] = agent.layers[0].policy.get_Q_values_pi(o[i, j, :], g, np.array([0, 0, 0, 0]), use_target_net=False)

                if agent.hparams["layers"] > 1:
                    g = np.array([1.15, 0.6, 0.5])
                    Q_vals_layer_1 = np.ones((10, 14))
                    o = np.array([1.30, 0.8, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    u = np.empty((10, 14, 3))

                    for i in range(10):
                        for j in range(14):
                            u[i, j, :] = np.array([1.075 + i*0.05, 0.425 +j*0.05, 0.5])
                            if agent.layers[1].policy is not None:
                                Q_vals_layer_1[i, j] = agent.layers[1].policy.get_Q_values_u(o, g, u[i, j, :], use_target_net=False)
                            elif agent.layers[1].critic is not None:
                                Q_vals_layer_1[i, j] = agent.layers[1].critic.get_Q_value(np.reshape(o,(1,10)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))


            # - - - - Q-vals for FetchPush - - - - 
            elif env.name == "FetchPush-v1" and agent.hparams["layers"] > 1:
                g = np.array([1.15, 0.6, 0.5])
                o = np.array([1.5, 1.0, 0.45,  1.4, 0.9, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                u = np.empty((10, 14, 3))

                for i in range(10):
                    for j in range(14):
                        u[i, j, :] = np.array([1.075 + i*0.05, 0.425 +j*0.05, 0.5])
                        if agent.layers[1].policy is not None:
                            Q_vals_layer_1[i, j] = agent.layers[1].policy.get_Q_values_u(o, g, u[i, j, :], use_target_net=False)
                        elif agent.layers[1].critic is not None:
                            Q_vals_layer_1[i, j] = agent.layers[1].critic.get_Q_value(np.reshape(o,(1,25)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))



            if batch == 5:
                Q_val_table[0, 0, :, :] = Q_vals_layer_0
                Q_val_table[0, 1, :, :] = Q_vals_layer_1
            elif batch == NUM_BATCH-1:
                Q_val_table[1, 0, :, :] = Q_vals_layer_0
                Q_val_table[1, 1, :, :] = Q_vals_layer_1







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


    return np.copy(success_rate_plt), np.copy(Q_val_table), np.copy(critic_loss_layer0), np.copy(critic_loss_layer1)