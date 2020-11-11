"""
Base class for defining agents (TD3 and DDPG extend this)
Author: Nick LaFarge
"""

from abc import ABC, abstractmethod
import tensorflow as tf
import os
import numpy as np
import logging

class GymAgent(ABC):
    """
    Base class to extend for defining agents for use with OpenAI Gym environments
    """

    def __init__(self, state_size, action_size, save_filename, nn_name='network', **kwargs):
        """

        :param state_size: Size of the state signal from the learning environment
        :param action_size: Dimension of the environment action
        :param save_filename: directory where to save the network (pathlib.Path object)
        :param nn_name: Name of neural network for saving
        """
        self.state_size = state_size
        self.action_size = action_size
        self.save_filename = save_filename
        self.nn_name = nn_name

        self.critic_losses = None
        self.actor_losses = None
        self.updates = None

    @abstractmethod
    def act(self, state):
        """
        Defines how the agent selects an action based on an observed state form the environment.

        :param state: Current environmental state (or observation)
        :return: Action sampled from given agent
        """
        pass

    @abstractmethod
    def on_t_update(self, old_state, action, new_state, reward, done):
        """
        Called every time an agent-environment interaction occurs

        :param old_state: State prior to action
        :param action: Action chosen by the actor network
        :param new_state: New state produced by the environment as a result of the given action
        :param reward: Reward given at the new state
        :param done: True if the episode is complete
        """
        pass

    @abstractmethod
    def on_episode_complete(self, episode_number):
        """
        Called every time an episode is complete

        :param episode_number: The number episode that just completed
        """

        pass

    def _setup_saver(self, sess, restore_from_file, saved_episode_number):
        """
        Set up the tensorflow saver, including the current episode number

        :param sess: tensorflow session
        :param restore_from_file: true if we want to load a saved network from a file
        :param saved_episode_number: if restore_from_file is true, this allows for resuming a particular episode
        :return: True if the networks was restored from a saved location
        """

        # Setup variables for saving/resuming
        with tf.compat.v1.variable_scope('counter'):
            self.episode_counter = tf.Variable(0, name='episode_counter')
            self.episode_counter_ph = tf.compat.v1.placeholder(tf.int32, name='episode_counter_ph')
            self.update_episode_number = self.episode_counter.assign(self.episode_counter_ph)

        # In case we want to save part of the way through.
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)  # Keep ALL of the saved agents

        restore_successful = False

        # Restore from file
        if restore_from_file:
            restore_successful = self.restore(sess, episode_number=saved_episode_number)

        # If we did not restore a saved version, the initialize variables
        if not restore_successful:
            sess.run(tf.compat.v1.global_variables_initializer())

        return restore_successful

    def save(self, sess, global_step=0):
        """
        Save the tensorflow graph

        :param sess: tensorflow session
        :param global_step: number to identify this save (typically episode number)
        """
        sess.run(self.update_episode_number,
                      feed_dict={self.episode_counter_ph: self.current_episode_number})
        filedir = str(self.save_filename / self.nn_name )
        self.saver.save(sess, filedir, global_step=global_step)

    def restore(self, sess, episode_number=False):
        """
        Restore a tensorflow graph

        :param sess: tensorflow session
        :param episode_number: episode number to load. If False, load the most recent episode
        """
        if not self.save_filename.exists():
            return False

        file_names = os.listdir(str(self.save_filename))
        file_names = [x for x in file_names if '.meta' in x]
        numbers = [int(x[8:x.index('.meta')]) for x in file_names]

        if not numbers:
            logging.warning(f'Could not restore agent at episode {episode_number}')
            return False

        if not episode_number:
            episode_number = np.max(numbers)

        checkpoint_path = self.save_filename / f"{self.nn_name}-{episode_number}"
        new_saver = tf.compat.v1.train.import_meta_graph(str(checkpoint_path) + '.meta')
        new_saver.restore(sess, str(checkpoint_path))

        self.current_episode_number = episode_number

        return True
