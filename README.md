# Master_thesis
This repository contains the code for the Master thesis implementation of the HAC algorithm. The folder fetch_environments contains the implementation for the OpenAI Gym fetch environments. If other environments are introduced, seperate folders will be created. When results for the environment are available I will publish them in a results folder inside the environment directory.



## Update Log

### Currently - Adding tool for all needed plots
(Success rate plots, Plotting reward, Plotting critic losses, Generating Q-function figures)

### 3/24/2020 - Parallelization of the code
Code can be executed in parallel to enable more efficient computation on a server.

### 3/18/2020 - Modularized impelmentation
Hyperparameters can be used to choose different modules (actorcritic/ ddpg) for each layer and configure these for each layer individually. 