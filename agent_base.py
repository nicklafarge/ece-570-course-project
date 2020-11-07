from abc import ABC, abstractmethod


class GymAgent(ABC):

    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

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
