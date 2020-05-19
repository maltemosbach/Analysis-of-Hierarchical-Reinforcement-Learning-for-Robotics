from design_agent_and_env import design_agent_and_env
from agent import Agent
from create_run import create_run
from tensorboardX import SummaryWriter
import tensorflow as tf
import os
import shutil

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def init_runs(date, hparams, num_runs, data_dir, FLAGS, NUM_BATCH, save_models, idx_run=0):
    """Script that initializes the runs and organizes the data for multiprocessing.
        Args:
            date (str): timestamp
            hparams: hyperparameters from run.py
            num_runs (int): number of consecutive runs this script should execute
            data_dir (str): path to data directory
            FLAGS: flags determining how the algorithm is run
            NUM_BATCH (int): total number of batches to be run
            save_models (bool): whether the trained agents should be saved
            idx_run (int): index used to organize the data
        """

    for m in range(num_runs):

        if num_runs > 1:
            idx_run = m


        # Directories for models and logging
        hp_dir = "/"
        for arg in hparams:
            if arg != "run":
                hp_dir = hp_dir + arg + "=" + str(hparams[arg]) + "/"
        hp_dir = hp_dir
        logdir = "./tb/" + date + hp_dir + "run_" + str(idx_run)
        modeldir = "./saved_agents/" + hp_dir + "models" + "_" + str(idx_run)

        sess = tf.compat.v1.InteractiveSession()
        writer_graph = tf.compat.v1.summary.FileWriter(logdir)
        writer = SummaryWriter(logdir)


        # Instantiate the agent and environment
        agent, env = design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams)

        # Begin training
        success_rates_run, Q_val_table_run, critic_loss_layer0, critic_loss_layer1 = create_run(FLAGS,env,agent,writer,sess, NUM_BATCH)

        if FLAGS.retrain:
            np.save(data_dir + "/sr_run_" + str(idx_run) + ".npy", success_rates_run)
            np.save(data_dir + "/Q_val_table_run_" + str(idx_run) + ".npy", Q_val_table_run)
            np.save(data_dir + "/critic_loss_layer0_run_" + str(idx_run) + ".npy", critic_loss_layer0)
            np.save(data_dir + "/critic_loss_layer1_run_" + str(idx_run) + ".npy", critic_loss_layer1)
            if m == 0:
                with open(data_dir + "/title.txt", 'w') as outfile:
                    outfile.write(hparams["env"])
                with open(data_dir + "/hparams.txt", 'w') as outfile:
                    outfile.write(str(hparams))


        # Saving models 
        if save_models:
            shutil.move("./models", modeldir)


        sess.close()
        tf.compat.v1.reset_default_graph()

    
