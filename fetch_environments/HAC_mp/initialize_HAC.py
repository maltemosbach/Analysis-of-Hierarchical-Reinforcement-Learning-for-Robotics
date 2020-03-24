from design_agent_and_env import design_agent_and_env
from agent import Agent
from run_HAC import run_HAC
from tensorboardX import SummaryWriter
import tensorflow as tf
import os
import shutil

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def init_HAC(date, hparams, num_runs, data_dir, FLAGS, NUM_BATCH, save_models, idx_run=1):

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
        success_rates_run = run_HAC(FLAGS,env,agent,writer,sess, NUM_BATCH)

        if FLAGS.retrain:
            np.save(data_dir + "/sr_run_" + str(idx_run) + ".npy", success_rates_run)
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

    
