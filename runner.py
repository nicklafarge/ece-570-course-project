"""
Runner script for training TD3 or DDPG agents
Author: Nick LaFarge
"""

import gym
import matplotlib.pyplot as plt
from TD3 import Td3Agent
from DDPG import DdpgAgent
import numpy as np
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from pathlib import Path
import pickle


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
    """
    Run the reinforcement learning process (either can be in deterministic or training mode)

    :param env_name: name of the environment to use for training
    :param agent_class: class that can be used to create an agent object
    :param max_episodes: number of episodes to use for training
    :param n_deterministic_episodes: number of deterministic episodes to test the controller
    :param output_freq: Frequency at which to output episode results
    :param env_seed: random seed for the environment (to control the start conditions)
    :param tf_seed: random seed for tensorflow (to control network initialization)
    :param max_time: maximum number of time steps before terminating an episode
    :param video_save_root: directory to specify video save location
    :param kwargs: passed to the agent's __init__ method
    :return: trained agent and data from run
    """

    # Create the OpaenAI gym environment
    env = gym.make(env_name)

    # Set up video saving directroy
    if video_save_root and not video_save_root.exists():
        video_save_root.mkdir()

    # Implement random seeds
    if env_seed:
        env.seed(env_seed)

    if tf_seed:
        tf.compat.v1.random.set_random_seed(tf_seed)

    # Find state and action sizes of the environment
    state_size = env.observation_space.shape[0]
    if not hasattr(env.action_space, 'n'):
        action_size = env.action_space.shape[0]
    else:
        action_size = env.action_space.n

    # Create agent
    agent = agent_class(state_size, action_size,
                        actor_limit=env.action_space.high[0],  # size of action (assumes all actions scaled the same)
                        **kwargs)

    # Initialize data lists for episodes
    return_info = dict(score_list=[],
                       deterministic_runs=[],
                       states=[],
                       actions=[],
                       rewards=[],
                       dones=[])

    # Iterate through episodes
    for episode in range(max_episodes + n_deterministic_episodes):

        # Reset environment for beginning of episode
        state = env.reset()

        # Determine if the action should be stochastic or not (only true during training)
        deterministic_action = episode >= max_episodes

        # Sum of reward over an episode
        episode_rewards_sum = 0

        # Episode data lists
        states = []
        rewards = []
        actions = []
        dones = []

        # Set up video recorder if we want a video of this episode
        video_recorder = None
        if deterministic_action and video_save_root:
            video_path = str(video_save_root / f'episode-{agent.current_episode_number}.mp4')
            video_recorder = VideoRecorder(env, video_path, enabled=video_save_root is not None)

        # Conduct episode
        done = False
        t = 0
        while not done:

            # Show the frame and save the movie frame if this is in testing mode
            if deterministic_action:
                env.render()
                video_recorder.capture_frame()

            # Get action from agent
            action = agent.act(state, deterministic=deterministic_action)

            # Get signals from environment
            new_state, reward, done, info = env.step(action)

            # Save data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            episode_rewards_sum += reward

            # enforce maximum time for episode
            done = done or t == max_time - 1

            # Tell the agent a step in the environment has occured
            if not deterministic_action:
                agent.on_t_update(state, action, new_state, reward, done)

            # Update state value
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

                if deterministic_action:
                    # Save episode reward
                    return_info['deterministic_runs'].append(episode_rewards_sum)

                    # Close the video recorder
                    video_recorder.close()
                    video_recorder.enabled = False
                else:
                    # Save episode reward
                    return_info['score_list'].append(episode_rewards_sum)

                    # Inform the agent that an episode has completed
                    agent.on_episode_complete(episode)

                # Update data lists
                return_info['states'].append(states)
                return_info['actions'].append(actions)
                return_info['rewards'].append(rewards)
                return_info['dones'].append(dones)

            # Update time
            t += 1

    # Save the agent if training has occurred
    if max_episodes > 0:
        agent.save(agent.sess, global_step=max_episodes)

    return agent, return_info


def train_agent(env_name, agent_class, restore_from_file=False, **kwargs):
    """
    Train an agent on the given environment

    :param env_name: name of the environment to use for training
    :param agent_class: class that can be used to create an agent object
    :param restore_from_file: true if we want to restore a saved state of the agent
    :param kwargs: passed to run_learner
    :return:
    """

    # Construct save locations
    agent_save = Path('.save') / agent_class.__name__
    save_location = agent_save / env_name

    # Create directories if they don't already exist
    if not agent_save.exists():
        agent_save.mkdir()

    if not save_location.exists():
        save_location.mkdir()

    # kwargs to specify file saving locations
    kwargs = dict(
        save_filename=save_location / 'network',
        video_save_root=save_location / 'animations',
        restore_from_file=restore_from_file,
        **kwargs
    )

    # Run the learner script
    agent, info = run_learner(env_name, agent_class, **kwargs)

    # Save data in a pickle file (rewards, losses, etc)
    data_location = str(save_location / 'training_data')

    # If we are restoring a saved model, we can load the saved data as well
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


# Here are the available environments that can be trained for TD3 and DDPG. Futher information can
# be found at: https://github.com/openai/gym/wiki/Table-of-environments
envs_list = [
    'Pendulum-v0',
    'BipedalWalker-v2',
    'BipedalWalkerHardcore-v3',
    'LunarLanderContinuous-v2',
    'HalfCheetah-v2',
    'Humanoid-v2',
    'Hopper-v2',
    'Walker2d-v2',
    'Ant-v2',
    'Reacher-v2',
    'InvertedPendulum-v2',
    'InvertedDoublePendulum-v2'
]

if __name__ == '__main__':

    # Specify which environment to run
    # env_name = 'LunarLanderContinuous-v2'
    # env_name = 'BipedalWalker-v3'
    # env_name = 'Pendulum-v0'
    env_name = 'InvertedDoublePendulum-v2'

    # Specify which agent to use (TD3 or DDPG)
    agent_class = Td3Agent
    # agent_class = DdpgAgent

    # True if we want to train, false if we want to load a saved model
    train_mode = False

    # For training
    if train_mode:
        agent, data = train_agent(env_name,
                                  agent_class,
                                  max_time=500,
                                  save_frequency=100,
                                  max_episodes=6000,
                                  n_deterministic_episodes=0,
                                  restore_from_file=False,
                                  )

    # For evaluation
    else:
        agent, data = train_agent(env_name,
                                  agent_class,
                                  max_time=1500,
                                  save_frequency=np.pi,  # A little hacky, but ensures it doesn't re-save the network
                                  max_episodes=0,
                                  n_deterministic_episodes=1,  # number of episodes to simulate and save
                                  restore_from_file=True,
                                  # episode_number=1500,
                                  # env_seed=4563,
                                  )
