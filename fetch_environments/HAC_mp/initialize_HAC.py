"""
This is the starting file for the Hierarchical Actor-Critc (HAC) modulesrithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

from design_agent_and_env import design_agent_and_env
from options import parse_options
from agent import Agent
from run_HAC import run_HAC
from tensorboardX import SummaryWriter
from datetime import datetime
import tensorflow as tf
import json
import time
import os
import shutil
from pathlib import Path
from utils import get_combinations

import argparse


#from utils import create_plot

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)





def init_HAC(date, hparams, num_runs, graph_number):

    print("In init_HAC!")

    # For matplotlib
    sr_array = []


    # Other options
    save_model = True




    for m in range(num_runs):
        print("Run number {no_run} of {total_runs}:".format(no_run=m+1, total_runs=int(num_runs)))
        print("Running with hyperparameters ", hparams)
        hp_dir = "/"
        for arg in hparams:
            if arg != "run":
                hp_dir = hp_dir + arg + "=" + str(hparams[arg]) + "/"
        hp_dir = hp_dir + "_" + str(m)

        logdir = "./tb/" + date + hp_dir
        modeldir = "./saved_agents/" + hp_dir + "models" + "_" + str(m)
        datadir = "./data/" + date
        Path(datadir).mkdir(parents=True, exist_ok=True)

        sess = tf.compat.v1.InteractiveSession()
        writer_graph = tf.compat.v1.summary.FileWriter(logdir)
        writer = SummaryWriter(logdir)

        with open(logdir + '/hparams.txt', 'w') as file:
             file.write(json.dumps(hparams))


        # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
        #FLAGS = parse_options()
        FLAGS = Namespace(HER=False, Q_values=False, all_trans=False, hind_action=False, penalty=False, play=False, prelim_HER=False, retrain=True, show=False, tensorboard=False, test=False, train_only=False, transfer=False, verbose=False)

        # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
        agent, env = design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams)

        # Begin training

        sr_array_tmp = run_HAC(FLAGS,env,agent,writer,sess)
        sr_array.append(sr_array_tmp)


        # If all runs for one combination are finished a new sr_array for the next combination is introduced
        if m >= num_runs-1:
            np.save(datadir + "/graph_" + str(graph_number), sr_array)
            with open(datadir + "/title.txt", 'w') as outfile:
                outfile.write(hparams["env"])
            with open(datadir + "/hp_graph.txt", 'w') as outfile:
                outfile.write("graph_" + str(graph_number) + ": " + str(hparams))


        # Save models 
        if save_model:
            shutil.move("./models", modeldir)


        sess.close()
        tf.compat.v1.reset_default_graph()

    
