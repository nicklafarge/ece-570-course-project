import gym
import matplotlib.pyplot as plt
from TD3 import Td3Agent
import numpy as np
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def run_learner(env_name,
                agent_class,
                max_episodes=1000,
                n_greedy_episodes=10,
                output_freq=10,
                env_seed=None,
                tf_seed=None,
                max_time=100,
                **kwargs):
    env = gym.make(env_name)

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
    score_list = []
    greedy_runs = []

    # Iterate through episodes
    for episode in range(max_episodes + n_greedy_episodes):
        state = env.reset()

        deterministic_action = episode >= max_episodes

        episode_rewards_sum = 0

        video_recorder = None

        if deterministic_action:
            video_path = f'.save/{env_name}-ppo-greedy-{episode}.mp4'
            video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

        done = False
        t = 0
        while not done:

            if deterministic_action:
                env.render()
                video_recorder.capture_frame()

            # tf.random.set_random_seed(episode*231)
            action = agent.act(state, deterministic=deterministic_action)

            new_state, reward, done, info = env.step(action)

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
                    avg_reward = np.average(score_list[-100:])
                    print(f"Average reward over 100 episodes: {avg_reward}")

                # Train agent
                if deterministic_action:
                    greedy_runs.append(episode_rewards_sum)
                    video_recorder.close()
                    video_recorder.enabled = False
                else:
                    score_list.append(episode_rewards_sum)
                    agent.on_episode_complete(episode)

            t += 1

    print(len(score_list))
    agent.save()

    return agent, score_list, greedy_runs


def compute_rolling_average(scores, N):
    return np.convolve(scores, np.ones((N,)) / N, mode='valid')


def run_agent():
    # env_name = 'CartPole-v1'
    # env_name = 'MountainCarContinuous-v0'
    # env_name = 'Pendulum-v0'
    # env_name = 'BipedalWalker-v3'
    # env_name = 'LunarLanderContinuous-v2'
    # env_name = 'HalfCheetah-v2'
    env_name = 'Humanoid-v2'

    agent_class = Td3Agent

    n_greedy_episodes = 5
    max_episodes = 1000

    max_time = 200
    output_freq = 10

    a_scale = {'Pendulum-v0': 2,
               'BipedalWalker-v3': 1,
               'LunarLanderContinuous-v2': 1,
               'HalfCheetah-v2': 1,
               'Humanoid-v2': 1}

    kwargs = dict(
        max_episodes=max_episodes,
        max_time=max_time,
        n_greedy_episodes=n_greedy_episodes,
        output_freq=output_freq,
        actor_limit=a_scale[env_name]
    )

    agent, scores, greedy_scores = run_learner(env_name, agent_class, **kwargs)

    N_rolling = 100
    x_vals = range(len(scores))[N_rolling - 1:]

    plt.figure()
    plt.plot(compute_rolling_average(scores, N_rolling), c='r')
    plt.xlabel('Episode Number')
    plt.ylabel('Reward (Rolling Average)')
    plt.title(f'{env_name} Trial Results')
    # plt.hlines(195, x_vals[0], x_vals[-1], color='k')
    plt.show(block=False)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(compute_rolling_average(np.abs(agent.actor_losses), N_rolling), c='r')
    plt.title('Actor Critic Loss Functions')
    plt.ylabel('Actor Loss')

    plt.subplot(2, 1, 2)
    plt.plot(compute_rolling_average(np.abs(agent.critic_losses), N_rolling), c='r')
    plt.ylabel('Critic Loss')
    plt.xlabel('Episode Number')
    plt.show(block=False)

    return agent, scores, greedy_scores


agent, scores, greedy_scores = run_agent()
