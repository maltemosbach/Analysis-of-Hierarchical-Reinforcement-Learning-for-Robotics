# Master thesis
This repository contains the code for the Master thesis implementation of the HAC algorithm. The folder fetch_environments contains the implementation for the OpenAI Gym fetch environments. If other environments are introduced, seperate folders will be created. When results for the environment are available I will publish them in a results folder inside the environment directory.


## Update Log

### Currently - Finish plotting tool and create transfer learning testcase
- Success_rate plots are too noisy
- Building variations of the original gym envs

### 4/2/2020 - Introduce own "transitions_buffer"
Since the experience_buffer and the baselines replay_buffer were not quite sufficient for testing sampling strategies as I had intended, I introduced my own transitions_buffer.

### 3/25/2020 - Create_plots tool
Arrays for success rate and a Q-value table is saved form the runs and loaded by create_plots.py to create the success rate figures as well as figures used to visualize the learning of the critic.

### 3/24/2020 - Parallelization of the code
Code can be executed in parallel to enable more efficient computation on a server.

### 3/18/2020 - Modularized impelmentation
Hyperparameters can be used to choose different modules (actorcritic/ ddpg) for each layer and configure these for each layer individually. 