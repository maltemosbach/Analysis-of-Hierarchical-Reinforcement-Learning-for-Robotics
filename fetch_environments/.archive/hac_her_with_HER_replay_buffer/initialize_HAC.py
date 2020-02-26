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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

combinations = {
        "ac_n"       : [0.2],
        "sg_n"       : [0.1],
        "inp_norm"   : [True],
        "no_of_run"  : [0, 1, 2, 3, 4]
    }

hparams = [{}, {}, {}, {}, {}] 
for i in range(len(combinations["ac_n"])):
    for j in range(len(combinations["sg_n"])):
        for k in range(len(combinations["inp_norm"])):
            for l in range(len(combinations["no_of_run"])):
                hparams[k + j + i + l] = {
                    "ac_n"          : combinations["ac_n"][i],
                    "sg_n"          : combinations["sg_n"][j],
                    "inp_norm"      : combinations["inp_norm"][k],
                    "no_of_run"     : combinations["no_of_run"][l]
                    }


date = datetime.now().strftime("%d.%m-%H:%M")


#for m in range(len(hparams)):
for m in range(5):
    print("len(hparams):", len(hparams))
    print("Running with hyperparameters ", hparams[m])
    hp_dir = "/"
    for arg in hparams[m]:
        hp_dir = hp_dir + arg + "=" + str(hparams[m][arg]) + "/"

    logdir = "./tb/" + date + hp_dir
    # To set session as default
    sess = tf.compat.v1.Session().__enter__()
    writer_graph = tf.compat.v1.summary.FileWriter(logdir)
    writer = SummaryWriter(logdir)

    with open(logdir + '/hparams.txt', 'w') as file:
         file.write(json.dumps(hparams[m]))



    # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
    FLAGS = parse_options()

    # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
    agent, env = design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams[m])

    # Begin training
    run_HAC(FLAGS,env,agent,writer,sess)
    sess.close()
    
    tf.compat.v1.reset_default_graph()

    
