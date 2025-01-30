# Advanced Topics in Machine Learning: Paper Replication Project
Authors: Jonatan Bella, Alessia Berarducci, Tobias Erbacher, Jonas Knupp

This repository contains the codebase and LaTeX sources for the replication study of the paper "[Random Latent Exploration for Deep Reinforcement Learning](https://arxiv.org/pdf/2407.13755)". Mahankali et al. provide some code in their [GitHub](https://github.com/Improbable-AI/random-latent-exploration) repository. This study was conducted as part of the **Advanced Topics in Machine Learning** course at Università della Svizzera italiana in the autumn semester 2024.

For the FourRoom and Atari experiments, we heavily relied on [Weights and Biases](https://wandb.ai/) for experiment tracking. Thus, we suggest using it when running code for these environments.

The codebase is structured in the following directories:
* **four_room_grid_world**: Contains the Gymnasium FourRoom environment, the adopted PPO, NoisyNet, RND, and RLE algorithms for the FourRoom environment, and the code to run the experiments described in our report. In particular, we added visualization capabilities to the algorithms.
* **ATARI games**: Contains the code for the PPO, NoisyNet, RND, and RLE algorithms for the Atari environment that was copied from the Mahankali et al.
* **Isaac Gym**: Contains the code to evaluate RLE in the CartPole environment of IsaacLab. 
- **adaptive_von_mises**: Implements both queue-based and neural-based adaptive von Mises-Fisher algorithms for the Atari environment.
