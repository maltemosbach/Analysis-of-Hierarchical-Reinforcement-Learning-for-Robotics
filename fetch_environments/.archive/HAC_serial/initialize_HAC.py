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

#from utils import create_plot

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hyperparameters = {
        "env"          : ['FetchReach-v1'],
        "ac_n"         : [0.2],
        "sg_n"         : [0.1],
        "replay_k"     : [4],
        "layers"       : [1],
        "use_target"   : [[False, False]],
        "sg_test_perc" : [0.1],
        "use_rb"       : [[True, False], [False, False]],
        "modules"      : [["ddpg", "actorcritic"]],
        "run"          : [0, 1]
    }

hparams = get_combinations(hyperparameters)

date = datetime.now().strftime("%d.%m-%H:%M")



# For matplotlib
sr_array = []


# Other options
save_model = True
ind = 0




for m in range(len(hparams)):
    print("Run number {no_run} of {total_runs}:".format(no_run=m+1, total_runs=int(len(hparams))))
    print("Running with hyperparameters ", hparams[m])
    hp_dir = "/"
    for arg in hparams[m]:
        hp_dir = hp_dir + arg + "=" + str(hparams[m][arg]) + "/"

    logdir = "./tb/" + date + hp_dir
    modeldir = "./saved_agents/" + hp_dir + "models"
    datadir = "./data/" + date
    Path(datadir).mkdir(parents=True, exist_ok=True)

    sess = tf.compat.v1.InteractiveSession()
    writer_graph = tf.compat.v1.summary.FileWriter(logdir)
    writer = SummaryWriter(logdir)

    with open(logdir + '/hparams.txt', 'w') as file:
         file.write(json.dumps(hparams[m]))


    # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
    FLAGS = parse_options()

    # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
    agent, env = design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams[m])

    # Begin training

    sr_array_tmp = run_HAC(FLAGS,env,agent,writer,sess)
    sr_array.append(sr_array_tmp)


    # If all runs for one combination are finished a new sr_array for the next combination is introduced
    if np.mod(m + 1, len(hyperparameters["run"])) == 0:
        np.save(datadir + "/graph_" + str(ind), sr_array)
        with open(datadir + "/title.txt", 'w') as outfile:
            outfile.write(hparams[m]["env"])
        with open(datadir + "/hp_graph.txt", 'w') as outfile:
            outfile.write("graph_" + str(ind) + ": " + str(hparams[m]))


        ind += 1
        sr_array = []


    # Save models 
    if save_model:
        shutil.move("./models", modeldir)


    sess.close()
    tf.compat.v1.reset_default_graph()

    
