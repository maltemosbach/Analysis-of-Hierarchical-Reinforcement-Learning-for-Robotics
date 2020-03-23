import multiprocessing
from datetime import datetime
from initialize_HAC import init_HAC
from utils import get_combinations_mp
import time

date = datetime.now().strftime("%d.%m-%H:%M")

hyperparameters = {
        "env"          : ['FetchPush-v1'],
        "ac_n"         : [0.2],
        "sg_n"         : [0.1],
        "replay_k"     : [4],
        "layers"       : [1, 2],
        "use_target"   : [[False, False]],
        "sg_test_perc" : [0.1],
        "use_rb"       : [[False, False]],
        "modules"      : [["ddpg", "actorcritic"]],

    }



hparams = get_combinations_mp(hyperparameters)

print("hparams:", hparams)

num_processes = len(hparams)

print("Running {num_p} parallel processes ...".format(num_p=num_processes))
time.sleep(10)

assert num_processes <= 6, "More than six parallel processes!"


if __name__ == '__main__':
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=init_HAC, args=(date, hparams[i], 2, i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        



















































