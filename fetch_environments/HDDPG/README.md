# HAC + HER
This implementation uses a ddpg agent like the original HER one from OpenAI for the lowest layer and actor and critic implementations similar to the original HAC ones for all layers above.

## Hyperparameters and runs
A list of hyperparameters is given in the initialize_HAC.py file. The file will start runs with all possible hparam combinations and log the results to tensorboard.

## Start runs
To run on a different environment, the environment.py file must be changed such that Gym creates the desired env. The functions for executing actions, resetting, returning the state or doing visualization, do not need to be altered.

