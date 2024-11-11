# Advanced Topics in Machine Learning Project
This repository contains the code and latex for the replication study for the paper "[Random Latent Exploration for Deep Reinforcement Learning](https://arxiv.org/pdf/2407.13755)". The authors of the RLE paper provide some code on their [GitHub](https://github.com/Improbable-AI/random-latent-exploration/tree/mai) page.

## Four Rooms Environment 
The four rooms environment is in a 51x51 grid. There is a horizontal and a vertical wall, dividing the grid world into four rooms. The rooms are connected by 4 corridors. The horizontal corridors are located +-10 from the centre cell and the vertical corridors are located +-5 from the centre cell. Since the two walls occupy a row/column the number of states is 50x50+4. In the rendered environment the start state is red, the goal state is green, and the agent's location is a blue circle.
The four room environment was initially proposed in this [paper](https://www.sciencedirect.com/science/article/pii/S0004370299000521) (referenced in RLE).

### four_room_grid_world/env
The RLE authors did not provide any code for the four rooms problem. Therefore, we implemented the four room environment ourselves using the Open AI gymnasium library. 

### four_room_grid_world/algorithms
Contains the code for PPO, PPO RND, Noisy Net, and PPO RLE.
Note that the PPO, PPO RND, and Noise Net algorithms were not implemented by us from the ground up. Instead, existing code was used and changed so that the code works with the FourRoom environment. However, PPO RLE was implemented by ourselves starting from the PPO algorithm.
- [Proximal Policy Optimization (PPO)]() (adopted from [CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py))
- [Random Network Distillation (RND)](https://arxiv.org/pdf/1810.12894) (adopted from Isaac Gym ppo_rnd)
- [NoisyNet](https://arxiv.org/pdf/1706.10295) (adopted from atari ppo_noisy_net)
- [RLE](https://arxiv.org/pdf/2407.13755) (adopted from Isaac Gym ppo_rle)

## Visualization
The file _util/plot_util.py_ contains functions to create figures like in the RLE paper. The _plot_heatmap_ function plots a heatmap of the state visit counts and the _plot_trajectories_ plots the provided five trajectories in the FourRoom grid world.

## Good sources
PPO:
- https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf (page 16-19)
- https://ai.stackexchange.com/questions/37608/why-clip-the-ppo-objective-on-only-one-side