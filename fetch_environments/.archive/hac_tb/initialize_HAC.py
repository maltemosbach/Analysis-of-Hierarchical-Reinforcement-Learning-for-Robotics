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

combinations = {
        "ac_n"       : [0.05, 0.1, 0.2],
        "sg_n"       : [0.1],
        "inp_norm"   : [False, True]
    }

hparams = [{}, {}, {}, {}, {}, {}] 
for i in range(len(combinations["ac_n"])):
    for j in range(len(combinations["sg_n"])):
        for k in range(len(combinations["inp_norm"])):
            hparams[k + 2*j + 2*1*i] = {
                "ac_n"          : combinations["ac_n"][i],
                "sg_n"         : combinations["sg_n"][j],
                "inp_norm"   : combinations["inp_norm"][k]
                }


date = datetime.now().strftime("%Y%m%d-%H%M%S")


for m in range(len(hparams)):
    print("Running with hyperparameters ", hparams[m])
    hp_dir = "/"
    for arg in hparams[m]:
        hp_dir = hp_dir + arg + "=" + str(hparams[m][arg]) + "/"

    logdir = "./tb/" + date + hp_dir
    sess = tf.Session()
    writer_graph = tf.summary.FileWriter(logdir)
    writer = SummaryWriter(logdir)

    with open(logdir + '/hparams.txt', 'w') as file:
         file.write(json.dumps(hparams[m]))



    # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
    FLAGS = parse_options()

    # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
    agent, env = design_agent_and_env(FLAGS, writer, writer_graph, sess, hparams[m])

    # Begin training
    run_HAC(FLAGS,env,agent,writer,sess)
    
    tf.compat.v1.reset_default_graph()

    sess.close()
