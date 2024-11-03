# Advanced Topics in Machine Learning Project
Link to paper https://arxiv.org/pdf/2407.13755

Code from paper https://github.com/Improbable-AI/random-latent-exploration/tree/main.

**The env folder will be deleted soon because it relies on the gym library which is deprecated. Use the env_gymnasium library which relies on gymnasium.**

## Four Rooms Environment 
The four rooms environment is in a 51x51 grid. There is a horizontal and a vertical wall, dividing the grid world into 4 rooms. The rooms are connected by 4 holes, which are located at +-5 from the centre cell of the grid. Since the two walls occupy a row/column the number of states is 50x50+4. The start state is red, the goal state is green, and the agent's location is a blue circle.

Paper were four room problem was initially proposed (cited in the paper we replicate): https://www.sciencedirect.com/science/article/pii/S0004370299000521

### four_room_grid_world/env
The EnvTransformator is required because in the FourRoomGridWorld apart from the agent's position the start and goal state is also included in the state. However, our algorithms only require the agent's position. The EnvTransformator is a wrapper for the environment and only returns the agent's position as state. 

### four_room_grid_world/algorithms
The only algorithm that is at least somewhat working is the QLearning algorithm. The other algorithms do not even learn to avoid hitting the wall/grid border forever.

## Algorithms
- Proximal Policy Optimization (PPO) (from [PPO CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py))
- Random Network Distillation (RND) (from [RND CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), must be adapted from Atari)
- NoisyNet (from PPO CleanRL and https://github.com/Kaixhin/NoisyNet-A3C/, must be adapted from Atari)
- RLE (adopt from Isaac Gym RLE)

## Visualization
- Rollout of multiple trajectories from a policy trained with
RLE in the middle of the training (1.5 million timesteps), where
each color denotes a distinct trajectory. 
- State visitation counts of all the methods after training for 2.5M time steps without any task reward

## TODO
<<<<<<< HEAD
- Create a separate package for the environment?
-Need to ask about the combination they do on the paper for the feature network
-improve plotting capabilities
-add the other algos
-isaac still a problem to solve for rle
