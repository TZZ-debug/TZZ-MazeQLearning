import sys
import random

import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import numpy as np

from env import Env
from constants import *

# Initialize pygame
maze = Env()
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('arial', 12)
# Window title
pygame.display.set_caption('Maze')
# Window size
window = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))

state, reward, done = maze.reset()
maze.render(window)
pygame.display.flip()

actions = [0, 1, 2, 3]
Q = np.zeros((TILE * TILE, len(actions)))
episodes = 400
gamma = 0.9
lr = .85
rList = []

for i in range(episodes):
    state, reward, done = maze.reset()
    action = None
    rAll = 0
    steps = 0  # Initialize the number of steps
    while not done:
        # Elipson-greedy strategy
        e = 1.0 / ((i // 100) + 1)
        if np.random.rand(1) < e:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state, :])

        # Greedy strategy
        # action = np.argmax(Q[state, :]+ np.random.randn(1,4) / (i + 1))

        # Render the pygame window
        # maze.render(window)
        # pygame.display.flip()
        # pygame.time.Clock()

        new_state, reward, done = maze.move(action)  # Return new_state, reward, done
        # Update the Q table
        Q[state, action] = (1 - lr) * Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]))
        # Update the state
        state = new_state
        # Accumulate rewards
        rAll += reward
        steps += 1  # Increment the number of steps

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
    rList.append(rAll)
    print(i + 1, '/', episodes, 'Accumulated Reward:', rAll, 'Steps:', steps)

print("Final Q-Table Values")
print(Q)
plt.plot(rList)
plt.xlabel("Episodes")
plt.ylabel("Accumulated rewards")
plt.show()

while True:
    state, reward, done = maze.reset()
    steps = 0  # Initialize the number of steps
    while not done:
        action = np.argmax(Q[state, :])
        new_state, reward, done = maze.move(action)
        state = new_state
        steps += 1  # Increment the number of steps
        maze.render(window)
        steps_text = font.render(f"Steps: {steps}", True, (255, 255, 255))
        window.blit(steps_text, (10, 10))  # Display the number of steps in the top-left corner
        # Quit the pygame application
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
        pygame.display.flip()
        pygame.time.Clock().tick(2)