import gym
import matplotlib.pyplot as plt
from TD3 import Td3Agent
from DDPG import DdpgAgent
import numpy as np
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from pathlib import Path
import pickle
import sys
import itertools

# For running in parallel
if len(sys.argv) == 1:
    task_id = 1
    num_tasks = task_id
else:
    task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])


def run_learner(env_name,
                agent_class,
                max_episodes=1000,
                n_deterministic_episodes=10,
                output_freq=10,
                env_seed=None,
                tf_seed=None,
                max_time=200,
                video_save_root=None,
                **kwargs):
    env = gym.make(env_name)
    kwargs['actor_limit'] = env.action_space.high[0]

    if video_save_root and not video_save_root.exists():
        video_save_root.mkdir()

    # SEED EVERYTHING
    if env_seed:
        env.seed(env_seed)

    if tf_seed:
        tf.random.set_random_seed(tf_seed)

    state_size = env.observation_space.shape[0]
    if not hasattr(env.action_space, 'n'):
        action_size = env.action_space.shape[0]
    else:
        action_size = env.action_space.n

    agent = agent_class(state_size, action_size, **kwargs)

    # List of scores
    return_info = dict(score_list=[],
                       deterministic_runs=[],
                       states=[],
                       actions=[],
                       rewards=[],
                       dones=[])

    # Iterate through episodes
    for episode in range(max_episodes + n_deterministic_episodes):
        state = env.reset()

        deterministic_action = episode >= max_episodes

        episode_rewards_sum = 0

        video_recorder = None

        states = []
        rewards = []
        actions = []
        dones = []

        if deterministic_action and video_save_root:
            video_path = str(video_save_root / f'episode-{episode}.mp4')
            video_recorder = VideoRecorder(env, video_path, enabled=video_save_root is not None)

        done = False
        t = 0
        while not done:

            if deterministic_action:
                env.render()
                video_recorder.capture_frame()

            # tf.random.set_random_seed(episode*231)
            action = agent.act(state, deterministic=deterministic_action)

            new_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            # enforce maximum time for episode
            done = done or t == max_time - 1

            episode_rewards_sum += reward

            if not deterministic_action:
                agent.on_t_update(state, action, new_state, reward, done)

            state = new_state

            if done:
                # Output data
                if episode % output_freq == 0 or deterministic_action:
                    print(f"episode: {episode}/{max_episodes},     "
                          f"score: {episode_rewards_sum},     "
                          f"critic loss: {'-' if not agent.critic_losses else np.abs(agent.critic_losses[-1])},     "
                          f"actor loss: {'-' if not agent.actor_losses else np.abs(agent.actor_losses[-1])},     "
                          f"loss: {'-' if not agent.updates else np.abs(agent.updates[-1])},     "
                          f"greedy: {deterministic_action}")

                if episode > 0 and episode % 100 == 0:
                    avg_reward = np.average(return_info['score_list'][-100:])
                    print(f"Average reward over 100 episodes: {avg_reward}")

                # Train agent
                if deterministic_action:
                    return_info['deterministic_runs'].append(episode_rewards_sum)
                    video_recorder.close()
                    video_recorder.enabled = False
                else:
                    return_info['score_list'].append(episode_rewards_sum)
                    agent.on_episode_complete(episode)

                return_info['states'].append(states)
                return_info['actions'].append(actions)
                return_info['rewards'].append(rewards)
                return_info['dones'].append(dones)
            t += 1

    # agent.save(agent.sess, global_step=max_episodes)

    return agent, return_info


def train_agent(env_name, agent_class, restore_from_file=False, **kwargs):
    agent_save = Path('.save') / agent_class.__name__
    save_location = agent_save / env_name
    # if not agent_save.exists():
    #     agent_save.mkdir()

    if not save_location.exists():
        save_location.mkdir()

    kwargs = dict(
        save_filename=save_location / 'network',
        video_save_root=save_location / 'animations',
        restore_from_file=restore_from_file,
        **kwargs
    )

    agent, info = run_learner(env_name, agent_class, **kwargs)

    data_location = str(save_location / 'training_data')
    if restore_from_file:
        with open(data_location, 'rb') as fp:
            data = pickle.load(fp)
            data['eval_scores'] = info['deterministic_runs']
            data['states'] = info['states']
            data['actions'] = info['actions']
            data['rewards'] = info['rewards']
            data['dones'] = info['dones']
    else:
        data = dict(
            scores=info['score_list'],
            deterministic_scores=info['deterministic_runs'],
            actor_losses=agent.actor_losses,
            critic_losses=agent.critic_losses
        )

        with open(data_location, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return agent, data

    # https://github.com/openai/gym/wiki/Table-of-environments


envs = [
    # 'Pendulum-v0',
    # 'BipedalWalker-v2',
    # 'BipedalWalkerHardcore-v3',
    # 'LunarLanderContinuous-v2',
    'HalfCheetah-v2',
    # 'Humanoid-v2',
    # 'Hopper-v2',
    # 'Walker2d-v2',
    # 'Ant-v2',
    # 'Reacher-v2',
    # 'InvertedPendulum-v2',
    # 'InvertedDoublePendulum-v2'
]
agents = [
    Td3Agent,
    # DdpgAgent
]

options = list(itertools.product(envs, agents))
config = options[task_id - 1]
env_name = config[0]
agent_class = config[1]

agent, data = train_agent(env_name,
                          agent_class,
                          max_time=500,
                          # save_frequency=100,
                          max_episodes=6000,
                          # n_deterministic_episodes=0,
                          # restore_from_file=False,
                          # max_episodes=1,
                          n_deterministic_episodes=1,
                          restore_from_file=True,
                          save_frequency=np.pi,
                          # episode_number=6500,
                          )

# N_rolling = 10
#
# plt.figure()
# plt.plot(compute_rolling_average(data['scores'], N_rolling), c='r')
# plt.xlabel('Episode Number')
# plt.ylabel('Reward (Rolling Average)')
# plt.title(f'{env_name} Trial Results')
# # plt.hlines(195, x_vals[0], x_vals[-1], color='k')
# plt.show(block=False)

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(compute_rolling_average(np.abs(data['actor_losses']), N_rolling), c='r')
# plt.title('Actor Critic Loss Functions')
# plt.ylabel('Actor Loss')
#
# plt.subplot(2, 1, 2)
# plt.plot(compute_rolling_average(np.abs(data['critic_losses']), N_rolling), c='r')
# plt.ylabel('Critic Loss')
# plt.xlabel('Episode Number')
# plt.show(block=False)
