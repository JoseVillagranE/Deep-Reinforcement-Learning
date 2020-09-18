import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG import DDPG
from utils import *

env = NormalizedEnv(gym.make("Pendulum-v0"))
# env = gym.make("Pendulum-v0")

batch_size = 128
agent = DDPG(env, batch_size=batch_size)
noise = OUNoise(env.action_space)
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        next_state, reward, done, _ = env.step(action)
        agent.replay_memory.add_to_memory((state, action, reward, next_state, done))

        if agent.replay_memory.get_memory_size() > batch_size:
            agent.update()

        state = next_state
        episode_reward += reward

        if done:
            print(f"episode: {episode}, reward: {np.round(episode_reward, decimals=2)}, avg_reward: {np.mean(rewards[-10:])}")
            break
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))


plt.plot(rewards)
plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

np.save("rewards.npy", np.array(rewards))
np.save("avg_rewards.npy", np.array(avg_rewards))
