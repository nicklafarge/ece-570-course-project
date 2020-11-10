from abc import ABC, abstractmethod
import tensorflow as tf
import os
import numpy as np

class GymAgent(ABC):

    def __init__(self, state_size, action_size, save_filename, nn_name='network', **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.save_filename = save_filename
        self.nn_name = nn_name

        self.critic_losses = None
        self.actor_losses = None
        self.updates = None

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def on_t_update(self, old_state, action, new_state, reward, done):
        pass

    @abstractmethod
    def on_episode_complete(self, episode_number):
        pass

    def _setup_saver(self, sess, restore_from_file, saved_episode_number):

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

        if not restore_successful:
            sess.run(tf.compat.v1.global_variables_initializer())

        return restore_successful

    def save(self, sess, global_step=0):
        sess.run(self.update_episode_number,
                      feed_dict={self.episode_counter_ph: self.current_episode_number})
        filedir = str(self.save_filename / self.nn_name )
        self.saver.save(sess, filedir, global_step=global_step)

    def restore(self, sess, episode_number=False):
        if not self.save_filename.exists():
            return False

        file_names = os.listdir(str(self.save_filename))
        file_names = [x for x in file_names if '.meta' in x]
        numbers = [int(x[8:x.index('.meta')]) for x in file_names]

        if not numbers:
            return False

        if not episode_number:
            episode_number = np.max(numbers)

        checkpoint_path = self.save_filename / f"{self.nn_name}-{episode_number}"
        new_saver = tf.compat.v1.train.import_meta_graph(str(checkpoint_path) + '.meta')
        new_saver.restore(sess, str(checkpoint_path))

        self.current_episode_number = episode_number
        return True
