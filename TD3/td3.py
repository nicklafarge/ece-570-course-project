from agent_base import GymAgent
from TD3.models import Actor, Critic
from TD3.utils import get_vars, ReplayBuffer
import numpy as np
import tensorflow as tf


class Td3Agent(GymAgent):
    def __init__(self, state_size, action_size,
                 gamma=0.99,
                 pi_lr=1e-4,
                 q_lr=1e-3,
                 polyak=0.995,
                 steps_per_epoch=4000,
                 replay_size=int(1e6),
                 batch_size=100,
                 epochs=50,
                 n_initial_random_actions=10000,
                 n_initial_replay_steps=1000,
                 update_frequency=50,
                 actor_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 restore_from_file=None,
                 save_frequency=500,
                 actor_network_layer_sizes=(256, 256),
                 critic_network_layer_sizes=(256, 256),
                 actor_limit=1.0,
                 **kwargs
                 ):
        """
        Args:
            gamma: learning rate
            pi_lr: learning rate for actor network
            q_lr: learning rate for critic network
            polyak: target network update hyperparameter
            steps_per_epoch: number of state-action pairs per epoch
            epochs: Number of epochs for training
            replay_size: Max size of replay buffer
            batch_size: Minibatch size for SGD
            n_initial_random_actions: number of random steps before policy is used
            n_initial_replay_steps: number of steps before training begins (for replay buffer)
            update_frequency: how often to run update
            actor_noise: Gaussian noise std. dev. added to policy during training
            restore_from_file: True if agent should be restored from a saved checkpoint
            save_frequency: how often to save neural network
            actor_network_layer_sizes: layer sizes for actor nn
            critic_network_layer_sizes: layer sizes for critic nn
            actor_limit: limit in action size

        """
        super().__init__(state_size, action_size)
        self.gamma = gamma
        self.actor_lr = pi_lr
        self.critic_lr = q_lr
        self.polyak = polyak
        self.epochs = epochs
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.n_initial_random_actions = n_initial_random_actions
        self.n_initial_replay_steps = n_initial_replay_steps
        self.update_frequency = update_frequency
        self.actor_noise = actor_noise
        self.restore_from_file = restore_from_file
        self.save_frequency = save_frequency
        self.steps_per_epoch = steps_per_epoch
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.actor_network_layer_sizes = actor_network_layer_sizes
        self.critic_network_layer_sizes = critic_network_layer_sizes
        self.actor_limit = actor_limit

        self.critic_losses = []
        self.actor_losses = []

        self.total_step_counter = 0

        self.main_scope = 'main'
        self.target_scope = 'target'
        self.actor_scope = 'actor'
        self.critic1_scope = 'critic1'
        self.critic2_scope = 'critic2'

        tf.compat.v1.disable_eager_execution()

        self._setup_placeholders()

        akwargs = dict(learning_rate=self.actor_lr, layer_sizes=self.actor_network_layer_sizes)
        ckwargs = dict(learning_rate=self.critic_lr, layer_sizes=self.critic_network_layer_sizes)

        self.actor = Actor(self.x_ph, self.action_size, self.actor_limit, self.main_scope, self.actor_scope, **akwargs)
        self.critic1 = Critic(self.x_ph, self.main_scope, self.critic1_scope, **ckwargs)
        self.critic2 = Critic(self.x_ph, self.main_scope, self.critic2_scope, **ckwargs)

        self.actor_target = Actor(self.x2_ph, self.action_size, self.actor_limit,
                                  self.target_scope, self.actor_scope, **akwargs)

        self.critic1_target = Critic(self.x2_ph, self.target_scope, self.critic1_scope, **ckwargs)
        self.critic2_target = Critic(self.x2_ph, self.target_scope, self.critic2_scope, **ckwargs)

        self._build_main()
        self._build_target()
        self._build_optimizers()
        self._define_actions()
        self._init_target()

        self.replay_buffer = ReplayBuffer(self.state_size, self.action_size, self.replay_size)

        # tf.disable_eager_execution()
        # stay on your own core, greedy tensorflow!
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        self.sess = tf.compat.v1.InteractiveSession(config=session_conf)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.target_init)

    def on_t_update(self, old_state, action, new_state, reward, done):
        self.total_step_counter += 1
        self.replay_buffer.store(old_state, action, new_state, reward, done)

        # Only optimize after replay buffer is ready
        if self.total_step_counter < self.n_initial_replay_steps:
            return

        if self.total_step_counter % self.update_frequency == 0:
            self.optimize()

    def on_episode_complete(self, episode_number):
        pass

    def optimize(self):

        # q_loss = self.sess.run([self.Q_loss], feed_dict)
        # pi_loss = 0

        q_losses_batch = []
        pi_losses_batch = []
        for j in range(self.epochs):
            batch = self.replay_buffer.sample_batch(self.batch_size)

            a_grads = self.sess.run(self.q1_pi_action_grads, feed_dict={self.x_ph: batch['states']})[0]
            feed_dict = {self.x_ph: batch['states'],
                         self.x2_ph: batch['new_states'],
                         self.a_ph: batch['actions'],
                         self.r_ph: batch['rewards'],
                         self.d_ph: batch['dones']
                         }

            q_loss, _ = self.sess.run([self.critic_loss, self.update_critic], feed_dict)
            q_losses_batch.append(q_loss)

            if j % self.policy_delay == 0:
                pi_loss, _, _ = self.sess.run([self.actor_loss, self.update_actor, self.target_update], feed_dict)
                pi_losses_batch.append(pi_loss)

        self.critic_losses.append(np.average(q_losses_batch))
        self.actor_losses.append(np.average(pi_losses_batch))

    def act(self, state, deterministic=False):

        if len(self.replay_buffer) < self.n_initial_random_actions:
            action = np.squeeze(np.random.uniform(-self.actor_limit, self.actor_limit, (self.action_size, 1)))
            if self.action_size == 1:
                action = np.array([float(action)])
            return action

        feed_dict = {self.x_ph: np.reshape(state, (1, self.state_size))}
        action_fn = self.deterministic_action if deterministic else self.sample_action
        action = self.sess.run(action_fn, feed_dict=feed_dict)[0]

        return action

    def _init_target(self):
        # Polyak averaging for target variables
        self.target_update = tf.group([tf.compat.v1.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                       for v_main, v_targ in
                                       zip(get_vars(self.main_scope), get_vars(self.target_scope))])

        # Initializing targets to match main variables
        self.target_init = tf.group([tf.compat.v1.assign(v_targ, v_main)
                                     for v_main, v_targ in
                                     zip(get_vars(self.main_scope), get_vars(self.target_scope))])

    def _setup_placeholders(self):
        self.x_ph = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), 'state')
        self.x2_ph = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), 'new_state')
        self.a_ph = tf.compat.v1.placeholder(tf.float32, (None, self.action_size), 'action')
        self.action_grads_ph = tf.compat.v1.placeholder(tf.float32, (None, self.action_size), 'action_grads')
        self.r_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'reward')
        self.d_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'done')

    def _build_main(self):
        self.pi = self.actor.build_network()
        self.q1, self.q1_action_grads = self.critic1.build_network(self.a_ph)
        self.q2, self.q2_action_grads = self.critic2.build_network(self.a_ph)
        self.q1_pi, self.q1_pi_action_grads = self.critic1.build_network(self.pi, reuse=True)

    def _build_target(self):
        self.pi_target = self.actor_target.build_network()

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.compat.v1.random_normal(tf.shape(self.pi_target), stddev=self.target_noise)
        epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
        a2 = self.pi_target + epsilon
        a2 = tf.clip_by_value(a2, -self.actor_limit, self.actor_limit)

        self.q1_target, self.q1_target_action_grads = self.critic1_target.build_network(a2)
        self.q2_target, self.q2_target_action_grads = self.critic2_target.build_network(a2)

    def _define_actions(self):
        with tf.compat.v1.variable_scope('sample_action'):
            epsilon = self.actor_noise * np.random.randn(self.action_size)
            self.sample_action = self.pi + epsilon

        with tf.compat.v1.variable_scope('deterministic_action'):
            self.deterministic_action = self.pi

    def _build_optimizers(self):

        self.actor_loss, self.update_actor = self.actor.define_optimizers(self.pi, self.action_grads_ph, self.q1_pi)
        self.critic1_loss = self.critic1.define_optimizers(self.r_ph, self.gamma,
                                                           self.d_ph, self.q1,
                                                           self.q1_target, self.q2_target)
        self.critic2_loss = self.critic1.define_optimizers(self.r_ph, self.gamma,
                                                           self.d_ph, self.q2,
                                                           self.q1_target, self.q2_target)

        self.critic_loss = self.critic1_loss + self.critic2_loss
        self.update_critic = self.critic1.optimizer.minimize(self.critic_loss,
                                                             var_list=get_vars('main/critic'))

    def save(self):
        pass
