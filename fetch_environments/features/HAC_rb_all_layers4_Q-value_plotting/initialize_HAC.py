"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
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

from utils import create_plot, create_Q_value_figure

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

combinations = {
        "ac_n"         : [0.2],
        "sg_n"         : [0.1],
        "replay_k"     : [4],
        "layers"       : [1],
        "use_target"   : [False],
        "sg_test_perc" : [0.3],
        "use_rb"       : ["all"], # none, lowest or all rn
        "algo"         : ["ddpg"],
        "run"          : [0]
    }

hparams = [{}] * len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"])*len(combinations["ac_n"])

print('len(hparams):', len(hparams))

for i in range(len(combinations["ac_n"])):
    for j in range(len(combinations["sg_n"])):
        for k in range(len(combinations["replay_k"])):
            for l in range(len(combinations["layers"])):
                for m in range(len(combinations["use_target"])):
                    for n in range(len(combinations["sg_test_perc"])):
                        for o in range(len(combinations["use_rb"])):
                            for p in range(len(combinations["algo"])):
                                for q in range(len(combinations["run"])):
                                    hparams[i*len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"])*len(combinations["sg_n"]) + j*len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"])*len(combinations["replay_k"]) + k*len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"])*len(combinations["layers"]) + l*len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"])*len(combinations["use_target"]) + m*len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"])*len(combinations["sg_test_perc"]) + n*len(combinations["run"])*len(combinations["algo"])*len(combinations["use_rb"]) + o*len(combinations["run"])*len(combinations["algo"]) + p*len(combinations["run"]) + q ] = {
                                        "ac_n"          : combinations["ac_n"][i],
                                        "sg_n"          : combinations["sg_n"][j],
                                        "replay_k"      : combinations["replay_k"][k],
                                        "layers"        : combinations["layers"][l],
                                        "use_target"    : combinations["use_target"][m],
                                        "sg_test_perc"  : combinations["sg_test_perc"][n],
                                        "use_rb"        : combinations["use_rb"][o],
                                        "algo"          : combinations["algo"][p],
                                        "run"           : combinations["run"][q]

                                        }


date = datetime.now().strftime("%d.%m-%H:%M")



# For matplotlib
setups = []
sr_plt = []




for m in range(len(hparams)):
    print("Run number {no_run} of {total_runs}:".format(no_run=m+1, total_runs=int(len(hparams))))
    print("Running with hyperparameters ", hparams[m])
    hp_dir = "/"
    for arg in hparams[m]:
        hp_dir = hp_dir + arg + "=" + str(hparams[m][arg]) + "/"

    logdir = "./tb/" + date + hp_dir
    print("logdir:", logdir)
    modeldir = "./saved_agents/" + hp_dir + "models"

    #sess = tf.compat.v1.Session().__enter__()
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

    sr_tmp, Q_vals = run_HAC(FLAGS,env,agent,writer,sess)
    sr_plt.append(sr_tmp)
    print("m =", m)
    print("len(combinations[run]) =", len(combinations["run"]))
    print('np.mod(m + 1, len(combinations["run"])):', np.mod(m + 1, len(combinations["run"])))

    if np.mod(m + 1, len(combinations["run"])) == 0:
        setups.append(sr_plt)
        print("sr_plt:", sr_plt)
        sr_plt = []


    # Save models
    save_model = False
    if save_model:
        shutil.move("./models", modeldir)

    print("setups:", setups)

    sess.close()
    tf.compat.v1.reset_default_graph()

plot = True

if plot:
    create_plot(setups, date)
    create_Q_value_figure(Q_vals, date)


    
