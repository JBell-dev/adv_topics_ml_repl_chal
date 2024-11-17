# Advanced Topics in Machine Learning Project
This repository contains the code and latex for the replication study for the paper "[Random Latent Exploration for Deep Reinforcement Learning](https://arxiv.org/pdf/2407.13755)". The authors of the RLE paper provide some code on their [GitHub](https://github.com/Improbable-AI/random-latent-exploration) page.

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

## Relevant sources
PPO:
- https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf (page 16-19)
- https://ai.stackexchange.com/questions/37608/why-clip-the-ppo-objective-on-only-one-side
[GAE](https://arxiv.org/pdf/1506.02438)

## Clarifications for RLE Atari and GridWorld
This section contains clarifications about the RLE Atari implementation. Especially, with respect to things not mentioned in the RLE paper.

### Where is normalization applied?
- The state features are standardized [*](https://github.com/jonupp/adv_topics_ml_repl_chal/blob/09115467f32bbf14f2df165faee1e369bfdd20a1/ATARI%20games/ppo_rle.py#L356C13-L356C21)
- The intrinsic rewards are normalized by division by the L2 norm of the feature vector [*](https://github.com/jonupp/adv_topics_ml_repl_chal/blob/09115467f32bbf14f2df165faee1e369bfdd20a1/ATARI%20games/ppo_rle.py#L357C54-L357C82)
- The intrinsic and extrinsic rewards are normalized by division of the "running standard deviation" calculated on the forward filtered rewards. [*](https://github.com/jonupp/adv_topics_ml_repl_chal/blob/09115467f32bbf14f2df165faee1e369bfdd20a1/ATARI%20games/ppo_rle.py#L724)

### What exactly is plotted in the game score plot and related plots?
- In every iteration the average of the past 128 returns in any environment is plotted at the last global step of the iteration in the "charts/game_score" plot.
- The "charts/episode_return" contains the actual returns per global step. Every time an episode in any environment ends, the return is stored with the corresponding global step.

### Why is not the difference between estimated value (from NN) and actual value (i.e., rewards from episode) used to train the value NN?
--> GAE

### Which aspects are not (directly) described in the paper but can be seen in the code?
- GAE
- Reward normalization
- Splitting value network's head (in contrast, the paper indicates that we want to maximize the sum of the intrinsic and extrinsic rewards directly)
- For GridWorld some hyperparameters are missing (Feature Network Update Rate tau, Latent Vector Resample Frequency, Intrinsic Discount Rate)
