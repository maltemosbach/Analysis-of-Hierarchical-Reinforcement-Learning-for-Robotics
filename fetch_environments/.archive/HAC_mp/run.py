"""
This is the starting file for all runs. The command line options are being processed and the algorithm is executed accordingly.
"""
import multiprocessing
from datetime import datetime
from initialize_HAC import init_HAC
from utils import get_combinations
from options import parse_options
from pathlib import Path

FLAGS = parse_options()



#  #  #  #  #  D E F I N E    A L L    P A R A M E T E R S  #  #  #  #  #
  

""" 1. HYPERPARAMETERS
The key hyperparameters are:
    env (str): Environment the algorithm should be run on
    ac_n (float): Noise added to the actions during exploration
    sg_n (float): Noise added to the subgoals during exploration
    replay_k (int): Number of HER transitions per regular transition
    layers (int): Number of hierarchical layers in the algorithm (1, 2)
    use_target (array of booleans): Whether each layer should use target networks
    sg_test_perc (float): Percentage of subgoal testing transitions
    use_rb (array of booleans): Whether each layer should use the replay buffer or standard experience buffer
    modules (array of strs): Modules each layer should use (ddpg, actorcritic right now)
"""
hyperparameters = {
        "env"          : ['FetchReach-v1'],
        "ac_n"         : [0.2],
        "sg_n"         : [0.1],
        "replay_k"     : [4],
        "layers"       : [1, 2],
        "use_target"   : [[False, False]],
        "sg_test_perc" : [0.1],
        "use_rb"       : [[False, False]],
        "modules"      : [["ddpg", "actorcritic"]],

    }


""" 2. PARAMETERS FOR RUNS AND TIME-SCALES
Parameters for the runs
    NUM_RUNS (int): Number of runs for each hyperparameter combination
    NUM_BATCH (int): Total number of batches for each run (one batch is made up of 10 (during testing) or 100 (during exploration) full episodes)
    FLAGS.time_scale (int): Max sequence length in which each policy will specialize
    FLAGS>max_actions (int): Max number of atomic actions
"""
NUM_RUNS = 2
NUM_BATCH = 21

FLAGS.time_scale = 10
FLAGS.max_actions = 50


""" 3. ADDITIONAL OPTIONS
More settings
    save_models (boolean): Whether all models should be saved
"""
save_models = True


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #











hparams = get_combinations(hyperparameters)

if FLAGS.test == False and FLAGS.retrain == False:
    FLAGS.retrain = True

date = datetime.now().strftime("%d.%m-%H:%M")

# Data directory to coordinate the output-data of the different runs
datadir = "./data/" + date
Path(datadir).mkdir(parents=True, exist_ok=True)

for i in range(len(hparams)):
    Path(datadir + "/graph_" + str(i)).mkdir(parents=True, exist_ok=True)



# Testing a trained agent
if FLAGS.test:
    assert len(hparams) == 1, "To test a trained agent only one parameter configuration should be given"
    init_HAC(date, hparams[0], 1, None, FLAGS, NUM_BATCH, False)


# Training new agents
if FLAGS.retrain:

    # Run such that one process is used for one hparam combination
    if FLAGS.np == len(hparams):
        print("Running (retraining) with {num_proc} processes ...".format(num_proc=len(hparams)))
        if __name__ == '__main__':
            processes = []
            for i in range(len(hparams)):
                p = multiprocessing.Process(target=init_HAC, args=(date, hparams[i], NUM_RUNS, datadir + "/graph_" + str(i), FLAGS, NUM_BATCH, save_models, ))
                processes.append(p)
                p.start()
                
            for process in processes:
                process.join()


    # Run such that each individual run is a process
    elif FLAGS.np == len(hparams)*NUM_RUNS:
        print("Running (retraining) with {num_proc} processes ...".format(num_proc=len(hparams)*NUM_RUNS))
        if __name__ == '__main__':
            processes = []
            for i in range(len(hparams)):
                for j in range(NUM_RUNS):
                    p = multiprocessing.Process(target=init_HAC, args=(date, hparams[i], 1, datadir + "/graph_" + str(i), FLAGS, NUM_BATCH, save_models, j,))
                    processes.append(p)
                    p.start()
                
            for process in processes:
                process.join()

    # Run in serial execution
    else:
        if FLAGS.np is not None and FLAGS.np != 1:
            print("Possible number of processes are 1, {n1}, or {n2} but {n3} was given instead. Falling back on serial execution.".format(n1=len(hparams), n2=len(hparams)*NUM_RUNS, n3=FLAGS.np))
        print("Running (retraining) in serial execution ...")
        for i in range(len(hparams)):
            init_HAC(date, hparams[i], NUM_RUNS, datadir + "/graph_" + str(i), FLAGS, NUM_BATCH, save_models)
