"""
Model definitions for TD3
Author: Nick LaFarge (reimplemented from OpenAI Spinning Up - TD3)

See https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/td3/core.py for original model implementation
"""
import tensorflow as tf


class Model(object):
    """
    Base class to define an arbitrary model (actor or critic)
    """

    def __init__(self, scope, layer_sizes=(256, 256), learning_rate=1e-3):
        """
        :param scope: Tensorflow scope that contains the trainable variables
        :param layer_sizes: sizes of the hidden layers
        :param learning_rate: learning rate for Adam optimizer
        """
        self.scope = scope
        self.learning_rate = learning_rate
        self.h1, self.h2 = layer_sizes
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

    @property
    def trainable_vars(self):
        """
        :return: Trainable variables for this model
        """
        return tf.compat.v1.trainable_variables(self.scope)


class Critic(Model):
    """
    Critic model for TD3 algorithm. Since multiple critics exist, this class provides a convenient way to produce
    multiple critic networks with the same structure.
    """

    def __init__(self, state_ph,
                 outer_scope,
                 inner_scope='critic',
                 **kwargs
                 ):
        """
        :param state_ph: Tensorflow placeholder that contains the state signal
        :param outer_scope: Outer scope for trainable variables (typically main or target)
        :param inner_scope: Inner scope for trainable variables (to designate which critic)
        :param kwargs: Addition arguments passed to Model
        """
        super().__init__(f'{outer_scope}/{inner_scope}', **kwargs)
        self.state_ph = state_ph

    def build_network(self, action_source, **kwargs):
        """
        Build a critic model with the action a sample from a give action_source

        General network structure consistent with Spinning Up (Lines 33,35,37).
        The specific network implementation is different, but the resulting network structure should be the same.

        :param action_source: Source of the action for computing the estimated Q(s,a) state-action value
        :param kwargs: Passed to tf.compat.v1.variable_scope
        :return: Critic network and action gradients
        """
        with tf.compat.v1.variable_scope(self.scope, **kwargs):
            l1 = tf.compat.v1.layers.dense(tf.concat([self.state_ph, action_source], axis=-1), self.h1, tf.nn.relu)
            l2 = tf.compat.v1.layers.dense(l1, self.h2, tf.nn.relu)
            Q = tf.squeeze(tf.compat.v1.layers.dense(l2, 1), axis=1)

        # Action gradients for optimizer
        action_grads = tf.gradients(Q, action_source)

        # Return results
        return Q, action_grads

    def define_optimizer(self, r_ph, gamma, d_ph, q, q1_targ, q2_targ):
        """
        Define optimizer for the critic network.

        Note two state-action value targets are included to reduce the value overestimation bias. The lower of the two
        is used for optimization.

        :param r_ph: Placeholder for reward
        :param gamma: Discount factor
        :param d_ph: Done placeholder
        :param q: state-action value
        :param q1_targ: first target state-action value
        :param q2_targ: second target state-action value
        :return: loss as a function of the given placeholders
        """

        # NOTE: this is a key contribution of TD3. By taking the minimum value of the two target critic networks, we
        # reduce the overestimation bias present in DDPG. Loss function computation reimplemented from Spinning Up.
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)

        # Standard MSE loss between computed q and the backup value
        loss = tf.reduce_mean(tf.compat.v1.squared_difference(q, backup))
        return loss


class Actor(Model):
    """
    Actor model for TD3 algorithm. Since the actor and the target actor exists , this class provides a convenient way
    to produce multiple actor networks with the same structure.
    """

    def __init__(self, state_ph,
                 action_size,
                 actor_limit,
                 outer_scope,
                 inner_scope='actor',
                 **kwargs):
        """
        :param state_ph: Placeholder for the state (or observation) signal
        :param action_size: Action dimension (number of nodes to place for the actor output)
        :param actor_limit: Action size limit (scaled from [-1,1] to [-actor_limit, actor_limit]. NOTE: this assumes
                            all actions are scaled the same (may not be true for all environments)
        :param outer_scope: Outer scope for trainable variables (typically main or target)
        :param inner_scope: Inner scope for trainable variables (to designate which critic)
        :param kwargs: Addition arguments passed to Model
        """
        super().__init__(f'{outer_scope}/{inner_scope}', **kwargs)
        self.state_ph = state_ph
        self.action_size = action_size
        self.actor_limit = actor_limit

    def build_network(self, **kwargs):
        """
        Build an actor network

        General network structure consistent with Spinning Up (Line 31).
        The specific network implementation is different, but the resulting network structure should be the same.

        :param kwargs: Passed to tf.compat.v1.variable_scope
        :return: Actor network
        """
        with tf.compat.v1.variable_scope(self.scope, **kwargs):
            l1 = tf.compat.v1.layers.dense(self.state_ph, self.h1, tf.nn.relu)
            l2 = tf.compat.v1.layers.dense(l1, self.h2, tf.nn.relu)
            pi = self.actor_limit * tf.compat.v1.layers.dense(l2, self.action_size, tf.nn.tanh)
        return pi

    def define_optimizer(self, actor_network, action_grads, q_pi):
        """
        Define the optimizer for the actor network. Code is included below to compute the update step in multiple ways -
        useful for both debugging and better understanding the underlying algorithm

        :param actor_network: actor network to compute the gradients for in the second update step method (unused)
        :param action_grads: action gradients for the second update step method (unused)
        :param q_pi: value function neural network (want to maximize value, i.e. minimize negative expected value)
        :return: loss and update step for optimizer
        """

        # NOTE: This is an alternate way to compute the update step for the actor network. The gradient is non-trivial
        # to compute and pass between the correct network, so it is useful to compute it in multiple ways for both
        # debugging, and to better learn how the gradient is calculated.
        # Source: Google Research's implementation of TD3 and TensorLayer's implementation (see links below)
        #   - Google: https://github.com/google-research/google-research/blob/master/dac/ddpg_td3.py
        #   - TL: https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_TD3.py
        actor_gradients = tf.gradients(actor_network, self.trainable_vars, -action_grads)
        update_step2 = self.optimizer.apply_gradients(zip(actor_gradients, self.trainable_vars))  # not used

        # Update step: we can compute the update step more simply by leveraging tensorflow's variable scopes to keep
        # track of which params we wish to update (we don't want to update critic values by accident), and using the
        # stop_gradient function in tensorflow when computing the bellman backup for the critic (see Critic network)
        loss = -tf.reduce_mean(q_pi)
        update_step = self.optimizer.minimize(loss, var_list=self.trainable_vars)

        return loss, update_step
