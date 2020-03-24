# Master_thesis
This repository contains the code for the Master thesis implementation of the HAC algorithm. The folder fetch_environments contains the implementation for the OpenAI Gym fetch environments. If other environments are introduced, seperate folders will be created. When results for the environment are available I will publish them in a results folder inside the environment directory.

Right now this code is beeing updated to allow for parallelized execution to enable efficient computation on a server.


## Update Log

### Currently - Parallelization of the code

### 3/18/2020 - Modularized impelmentation
Hyperparameters can be used to choose different modules (actorcritic/ ddpg) for each layer and configure these for each layer individually. 