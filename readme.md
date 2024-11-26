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

## Clarifications in General and for RLE Atari and GridWorld
This section contains clarifications about the RLE Atari implementation. Especially, with respect to things not mentioned in the RLE paper.

### What are the authors comparing RLE to?
- **Proximal Policy Optimization**: PPO uses action noise exploration. This is, we get the logits over the actions from the policy network. Then the logits are fed into a softmax (aka Boltzmann distribution) to obtain a distribution over the actions. The next action is then sampled from this distribution.
- **Noisy Net**: Example from the family of noise-based exploration
- **Random Network Distillation**: Example from the family of bonus-based exploration

Note: Since all algorithms are built on top of PPO, all of them also use action noise exploration.

### Where is normalization applied?
- The state features are standardized [*](https://github.com/jonupp/adv_topics_ml_repl_chal/blob/09115467f32bbf14f2df165faee1e369bfdd20a1/ATARI%20games/ppo_rle.py#L356C13-L356C21)
- The intrinsic rewards are normalized by division by the L2 norm of the feature vector [*](https://github.com/jonupp/adv_topics_ml_repl_chal/blob/09115467f32bbf14f2df165faee1e369bfdd20a1/ATARI%20games/ppo_rle.py#L357C54-L357C82)
- The intrinsic and extrinsic rewards are normalized by division of the "running standard deviation" calculated on the forward filtered rewards. [*](https://github.com/jonupp/adv_topics_ml_repl_chal/blob/09115467f32bbf14f2df165faee1e369bfdd20a1/ATARI%20games/ppo_rle.py#L724)
- The advantage is normalized

### What exactly is plotted in the game score plot and related plots?
- In every iteration the average of the past 128 returns in any environment is plotted at the last global step of the iteration in the "charts/game_score" plot.
- The "charts/episode_return" contains the actual returns per global step. Every time an episode in any environment ends, the return is stored with the corresponding global step.

### With what is the log probability of the action multiplied in the policy gradient?
All the implementations use generalized advantage estimator (GAE). We have the TD residual of $V$ as $\delta^V_t = r_t + \gamma V(s_{t_1})-V(s_t)$. The GAE is then $\sum_{l=0}^\infty (\gamma \lambda)^l \delta^V_{t+l}$. The parameter $\lambda$ controls the weighting of the future TD-residuals. We have the two special cases: GAE($\gamma, 0$)=$r_t+\gamma V(s_{t+1})-V(s_t)$ (which is the TD residual) and GAE($\gamma, 1$)=$\sum_{l=0}^\infty \gamma^l r_{t_l} - V(s_t)$ (which is the advantage function). The advantage of using GAE is that one can control the trade-off between bias and variance: $\lambda=0$ has low variance but high bias while $\lambda=1$ has high variance but low bias. Note that in the code the GAE is calculated recursively. The derivation of this recursive GAE formula is similar to the derivation of the recursive formula for the discounted rewards.

### How is the value loss calculated?
For simplicity, we consider a batch of size 1. The most basic loss for the value network would be $0.5(V^{new}(s_t)-G_t)^2$ where $G_t$ is the return. However, in all algorithms they use $0.5(V^{new}(s_t)-(\text{advantage} + V^{old}(s_t)))^2$ where $V^{new}$ is the up-to-date value network and $V^{old}$ is the value network from the data collection phase. Apparently, this reduces variance and stabilizes the training.

### Which aspects are not (directly) described in the paper but can be seen in the code?
- In Atari the rewards are clipped
- Reward normalization
- Splitting value network's head (in contrast, the paper indicates that we want to maximize the sum of the intrinsic and extrinsic rewards directly)
- For GridWorld some hyperparameters are missing (Feature Network Update Rate tau, Latent Vector Resample Frequency, Intrinsic Discount Rate)
