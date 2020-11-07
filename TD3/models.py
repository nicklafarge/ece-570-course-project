import tensorflow as tf


class Model(object):

    def __init__(self, scope, layer_sizes=(400, 300), learning_rate=1e-3):
        self.scope = scope
        self.learning_rate = learning_rate
        self.h1, self.h2 = layer_sizes
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

    @property
    def trainable_vars(self):
        return tf.compat.v1.trainable_variables(self.scope)


class Critic(Model):

    def __init__(self, state_ph,
                 outer_scope,
                 inner_scope='critic',
                 **kwargs
                 ):
        super().__init__(f'{outer_scope}/{inner_scope}', **kwargs)
        self.state_ph = state_ph

    def build_network(self, action_source, **kwargs):
        with tf.compat.v1.variable_scope(self.scope, **kwargs):
            l1 = tf.compat.v1.layers.dense(tf.concat([self.state_ph, action_source], axis=-1), self.h1, tf.nn.relu)
            l2 = tf.compat.v1.layers.dense(l1, self.h2, tf.nn.relu)
            Q = tf.squeeze(tf.compat.v1.layers.dense(l2, 1), axis=1)

        action_grads = tf.gradients(Q, action_source)
        return Q, action_grads

    def define_optimizers(self, r_ph, gamma, d_ph, q, q1_targ, q2_targ):
        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)
        loss = tf.reduce_mean(tf.compat.v1.squared_difference(q, backup))
        return loss


class Actor(Model):

    def __init__(self, state_ph,
                 action_size,
                 actor_limit,
                 outer_scope,
                 inner_scope='actor',
                 **kwargs):
        super().__init__(f'{outer_scope}/{inner_scope}', **kwargs)
        self.state_ph = state_ph
        self.action_size = action_size
        self.actor_limit = actor_limit

    def build_network(self, **kwargs):
        with tf.compat.v1.variable_scope(self.scope, **kwargs):
            l1 = tf.compat.v1.layers.dense(self.state_ph, self.h1, tf.nn.relu)
            l2 = tf.compat.v1.layers.dense(l1, self.h2, tf.nn.relu)
            pi = self.actor_limit * tf.compat.v1.layers.dense(l2, self.action_size, tf.nn.tanh)
        return pi

    def define_optimizers(self, actor_network, action_grads, q_pi):
        loss = -tf.reduce_mean(q_pi)

        # actor_gradients = tf.gradients(actor_network, self.trainable_vars, -action_grads)
        # update_step = self.optimizer.apply_gradients(zip(actor_gradients, self.trainable_vars))
        update_step = self.optimizer.minimize(loss, var_list=self.trainable_vars)

        return loss, update_step
