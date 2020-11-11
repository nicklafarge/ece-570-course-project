"""
Scripts for visualizing reward over time for trained networks
"""
import matplotlib.pyplot as plt
from TD3 import Td3Agent
from DDPG import DdpgAgent
import numpy as np
from pathlib import Path
import pickle


def compute_rolling_average(scores, N):
    return np.convolve(scores, np.ones((N,)) / N, mode='valid')


def get_data(agent, env):
    root_path = Path('.save')
    agent_path = root_path / agent.__name__
    data_location = agent_path / env / 'training_data'
    with open(str(data_location), 'rb') as fp:
        training_data = pickle.load(fp)
    return training_data


def show_for_basic_envs():
    envs = [
        'Pendulum-v0',
        'BipedalWalker-v2',
        'BipedalWalkerHardcore-v2',
        'LunarLanderContinuous-v2',
        # 'HalfCheetah-v2',
        # 'Humanoid-v2',
        # 'Hopper-v2',
        # 'Walker2d-v2',
        # 'Ant-v2',
        # 'Reacher-v2',
        # 'InvertedPendulum-v2',
        # 'InvertedDoublePendulum-v2'
    ]

    n_roll = 300

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axs.flatten()):
        env = envs[i]

        ddpg_data = get_data(DdpgAgent, env)
        td3_data = get_data(Td3Agent, env)

        ddpg_scores = ddpg_data['scores']
        td3_scores = td3_data['scores']
        ax.plot(compute_rolling_average(ddpg_scores, n_roll), c='tab:red', label='DDPG')
        ax.plot(compute_rolling_average(td3_scores, n_roll), c='tab:blue', label='TD3')
        ax.set_title(env)
        ax.set_ylabel('Discounted Return')
        ax.set_xlabel('Training Epoch')

    axs[0][0].legend()
    plt.tight_layout()


plt.figure()
n_roll = 300
env = 'HalfCheetah-v2'
# agent = DdpgAgent
agent = Td3Agent
data = get_data(agent, env)
# plt.plot(compute_rolling_average(get_data(DdpgAgent, env)['scores'], n_roll), c='tab:red', label='DDPG')
plt.plot(compute_rolling_average(get_data(Td3Agent, env)['scores'], n_roll), c='tab:blue', label='TD3')
plt.legend()
plt.title(env)
plt.ylabel("Discounted Return")
plt.xlabel("Training Epoch")
plt.tight_layout()
plt.show()
