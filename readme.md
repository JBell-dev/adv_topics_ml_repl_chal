# Advanced Topics in Machine Learning Project
Link to paper https://arxiv.org/pdf/2407.13755

Code from paper https://github.com/Improbable-AI/random-latent-exploration/tree/main.

## Four Rooms Environment 
The four rooms environment is in a 51x51 grid. There is a horizontal and a vertical wall, dividing the grid world into 4 rooms. The rooms are connected by 4 holes, which are located at +-5 from the centre cell of the grid. Since the two walls occupy a row/column the number of states is 50x50+4. The start state is red, the goal state is green, and the agent's location is a blue circle.

Paper were four room problem was initially proposed (cited in the paper we replicate): https://www.sciencedirect.com/science/article/pii/S0004370299000521

### four_room_grid_world/env
The EnvTransformator is required because in the FourRoomGridWorld apart from the agent's position the start and goal state is also included in the state. However, our algorithms only require the agent's position. The EnvTransformator is a wrapper for the environment and only returns the agent's position as state. 

### four_room_grid_world/algorithms
The only algorithm that is at least somewhat working is the QLearning algorithm. The other algorithms do not even learn to avoid hitting the wall/grid border forever.

# TODO
- Create a separate package for the environment

