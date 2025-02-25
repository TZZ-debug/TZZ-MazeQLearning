# Reinforcement Learning: Maze Solving

## Project Overview
This project uses the Q-learning algorithm in reinforcement learning to solve the maze problem. The maze is randomly generated each time it runs, presenting different challenges for training and testing.

## Project Structure
The project contains the following main files:
- `constants.py`: Defines constant variables, such as sprite size, screen size, and the paths of various image files.
- `startTrain.py` and `startFinal.py`: Contain the main program code for the training and final display phases. The Pygame library is used for visualization, and the Q-learning algorithm is used to train the agent to find the optimal path in the maze.
- `readme.md`: The documentation of the project, introducing the basic information and usage methods of the project.
- `env.py`: Defines the `Env` class, which is responsible for the generation of the maze environment, the initialization and update of states, and rendering functions.

## Running Environment
- Python 3
- Numpy
- PyGame
- matplotlib

## Installation and Usage
1. **Clone the project**:
   ```sh
   git clone <Project Repository Address>
   cd <Project Directory>
   ```
2. **Install dependencies**:
   Make sure you have installed Python 3, and then use the following command to install the required Python libraries:
   ```sh
   pip install numpy pygame matplotlib
   ```
3. **Run the training and Result**:
   To start the training process, run the `startFinal.py` file:
   ```sh
   python startFianl.py
   ```
   The training process will be visualized in the Pygame window, and the accumulated reward and number of steps for each episode will be output. After the training is completed, a chart showing the change of accumulated rewards with the number of episodes will be plotted.

4. **Final display**:
   After the training is completed, the program will enter the final display phase. The agent will find the optimal path in the maze according to the learned Q-table. You can exit the program by closing the Pygame window or pressing the `ESC` key.

## Configuration Instructions
If you want to change the size of the maze, you can modify the value of the `TILE` constant in the `constants.py` file.

## Algorithm Principle
This project uses the Q-learning algorithm to train the agent to find the optimal path in the maze. Q-learning is a value function-based reinforcement learning algorithm that learns the optimal strategy by continuously updating the Q-table.

During the training process, the agent selects an action based on the current state. After executing the action, it enters a new state and receives the corresponding reward. Then, the Q-table is updated according to the Q-learning update formula:
```python
Q[state, action] = (1 - lr) * Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]))
```
where `lr` is the learning rate and `gamma` is the discount factor.

## Notes
- When running the program, please ensure that all image files (located in the `images` directory) exist; otherwise, the program may encounter errors.
- The training process may take a certain amount of time, depending on the size of the maze and the number of training episodes.