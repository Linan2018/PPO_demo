import tensorflow as tf
import numpy as np
from copy import deepcopy
import gym


class Buffer:
    """
    store {state, action, reward, gain}
    """
    def __init__(self):
        self.data = {
            's':   [],
            'a':   [],
            'r':   [],
            'g':   []
        }
        self.total_reward_record = []

    def put(self, data, name):
        self.data[name].extend(data)

    def get(self, name):
        return np.array(deepcopy(self.data[name]))

    def clear(self):
        self.data = {
            's':   [],
            'a':   [],
            'r':   [],
            'g':   []
        }


class PPO:
    def __init__(self, config):
        # env settings
        self.env = gym.make(config["env_name"]).unwrapped
        self.s_dim = config["s_dim"]
        self.a_dim = config["a_dim"]
        # sample settings
        self.max_episode = config["max_episode"]
        self.max_sample_step = config["max_sample_step"]
        self.discount = config["discount"]
        self.td = config["td"]
        # training settings
        self.epsilon = config["epsilon"]
        self.actor_lr = tf.train.exponential_decay(
            config["actor_lr"], tf.Variable(0, trainable=False), 400, 0.1, staircase=True)
        self.critic_lr = tf.train.exponential_decay(
            config["critic_lr"], tf.Variable(0, trainable=False), 200, 0.5, staircase=False)
        self.batch_size = config["batch_size"]
        self.max_update_step = config["max_update_step"]
        self.update_actor_step = config["update_actor_step"]

        self.buffer = Buffer()

        self.sess = tf.Session()
        self.state_ph = tf.placeholder(tf.float32, [None, self.s_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, self.a_dim])
        self.advantage_ph = tf.placeholder(tf.float32, [None, 1])
        self.discounted_reward_ph = tf.placeholder(tf.float32, [None, 1])

        # --------------------------------actor_net----------------------------------------
        self.action_distribution, theta = self._build_actor_net('actor_net', trainable=True)
        self.action_distribution_old, theta_old = self._build_actor_net('actor_net_old', trainable=False)

        self.c_action = tf.squeeze(self.action_distribution_old.sample(1), axis=0)

        self.update_theta_old = [p_old.assign(p) for p, p_old in zip(theta, theta_old)]

        # importance sampling ratio
        ratio = self.action_distribution.prob(self.action_ph) / (self.action_distribution_old.prob(self.action_ph) + 1e-5)

        # using PPO-Clip
        # loss = reduce_mean(min(ratio * A, clip(ratio, 1-e, 1+e) * A))
        self.actor_loss = - tf.reduce_mean(tf.minimum(
            ratio * self.advantage_ph,
            tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.advantage_ph))

        self.train_actor = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss)

        # --------------------------------critic_net----------------------------------------
        self.v = self._build_critic_net()
        self.adv = self.discounted_reward_ph - self.v

        self.critic_loss = tf.reduce_mean(tf.square(self.adv))
        self.train_critic = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        self.sess.run(tf.global_variables_initializer())

    def _build_actor_net(self, name, trainable):
        # continuous action
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state_ph, 256, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_critic_net(self):
        with tf.variable_scope("critic_net"):
            dense1 = tf.layers.dense(self.state_ph, 128, tf.nn.relu)
            v = tf.layers.dense(dense1, 1)
        return v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.c_action, {self.state_ph: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        v = self.sess.run(self.v, {self.state_ph: s})[0, 0]
        return v

    def sample_data(self):
        total_reward = 0
        temp_buffer_s, temp_buffer_a, temp_buffer_r, temp_buffer_g = [], [], [], []

        for i in range(self.max_episode):
            s = self.env.reset()
            for j in range(self.max_sample_step):
                a = self.choose_action(s)
                s_, r, done, _ = self.env.step(a)

                temp_buffer_s.append(s)
                temp_buffer_a.append(a)
                temp_buffer_r.append((r + 8) / 8)  # normalize reward

                s = s_
                total_reward += r

            list_g = []  # for each episode
            # TD method
            if self.td:
                v_s_ = self.get_v(s)
                for r in temp_buffer_r[-1:-(self.max_sample_step+1):-1]:
                    v_s_ = r + self.discount * v_s_
                    list_g.append(v_s_)
                list_g.reverse()

            # MC method
            else:
                n = self.max_sample_step - 1
                list_g = [0] * (n + 1)
                # g_t = r_{t+1} + discount * g_{t+1}
                t = n
                while t > 0:
                    list_g[t - 1] = temp_buffer_r[t] + self.discount * list_g[t]
                    t -= 1
            # print(i, len(list_g))
            temp_buffer_g.extend(deepcopy(list_g))

            del list_g

            # print(len(temp_buffer_s), len(temp_buffer_a), len(temp_buffer_g))
        self.buffer.put(deepcopy(temp_buffer_s), 's')
        self.buffer.put(deepcopy(temp_buffer_a), 'a')
        # self.buffer.put(deepcopy(temp_buffer_r), 'r')
        self.buffer.put(deepcopy(temp_buffer_g), 'g')

        print(total_reward)
        self.buffer.total_reward_record.append(total_reward)

    def update(self, it):
        if not it % self.update_actor_step:
            self.sess.run(self.update_theta_old)
        s, a, g = self.buffer.get('s'), self.buffer.get('a'), self.buffer.get('g')[:, np.newaxis]
        # print(s.shape, a.shape, g.shape)
        adv = self.sess.run(self.adv, {self.state_ph: s, self.discounted_reward_ph: g})

        n = s.shape[0]
        for _ in range(self.max_update_step):
            for i in range(n // self.batch_size):
                # update actor
                self.sess.run(self.train_actor, {
                    self.state_ph: s[i * self.batch_size: i * self.batch_size + self.batch_size],
                    self.action_ph: a[i * self.batch_size: i * self.batch_size + self.batch_size],
                    self.advantage_ph: adv[i * self.batch_size: i * self.batch_size + self.batch_size]})
                # update critic
                self.sess.run(self.train_critic, {
                    self.state_ph: s[i * self.batch_size: i * self.batch_size + self.batch_size],
                    self.discounted_reward_ph: g[i * self.batch_size: i * self.batch_size + self.batch_size]})
        self.buffer.clear()

    def get_record(self):
        return self.buffer.total_reward_record

    def run_env(self):
        while True:
            s1 = self.env.reset()
            for t in range(self.max_sample_step):
                self.env.render()
                s1 = self.env.step(self.choose_action(s1))[0]
