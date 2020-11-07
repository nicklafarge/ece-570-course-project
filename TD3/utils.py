import tensorflow as tf
import numpy as np


def get_vars(scope):
    return [x for x in tf.compat.v1.global_variables() if scope in x.name]


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.state_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.new_state_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.action_buffer = np.zeros([size, act_dim], dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, old_state, action, new_state, reward, done):
        self.state_buffer[self.ptr] = np.squeeze(old_state)
        self.action_buffer[self.ptr] = np.squeeze(action)
        self.new_state_buffer[self.ptr] = np.squeeze(new_state)
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(states=self.state_buffer[idxs],
                    new_states=self.new_state_buffer[idxs],
                    actions=self.action_buffer[idxs],
                    rewards=self.reward_buffer[idxs],
                    dones=self.done_buffer[idxs])

    def __len__(self):
        return self.size
