# -*- coding:utf-8 -*-
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
自编码器 无监督学习 在复原自己的过程中学习高维度的特征 其作用类似PCA
"""


def xavier_init(in_nodes, out_nodes, constant = 1):
    low = -constant * np.sqrt(6.0 / (in_nodes + out_nodes))
    high = constant * np.sqrt(6.0 / (in_nodes + out_nodes))

    return tf.random_uniform((in_nodes, out_nodes), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoEncoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer, scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initial_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(
            tf.matmul(self.x+scale*tf.random_normal((n_input,)), self.weights["w1"]), self.weights["b1"]))
        self.reconstruction = self.transfer(tf.add(tf.matmul(self.hidden, self.weights["w2"]), self.weights["b2"]))

        # 方差  subtract 逐个求x-y
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x, self.reconstruction), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _initial_weights(self):
        all_weights = dict()
        all_weights["w1"] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights["b1"] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights["w2"] = tf.Variable(xavier_init(self.n_hidden, self.n_input))
        all_weights["b2"] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, batch_data):
        cost, opt = self.session.run((self.cost, self.optimizer),
                                     feed_dict={self.x: batch_data, self.scale: self.training_scale})
        return [cost, opt]

    def calc_total_cost(self, data):
        cost = self.session.run(self.cost, feed_dict={self.x: data, self.scale:self.training_scale})

        return cost

    def transform(self, data):
        return self.session.run(self.hidden, feed_dict={self.x: data, self.scale:self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])

        return self.session.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, data):
        return self.session.run(self.reconstruction, feed_dict={self.x: data, self.scale: self.training_scale})

    def get_weight(self):
        return self.session.run(self.weights["w1"])

    def get_biases(self):
        return self.session.run(self.weights["b1"])


def standard_scale(x_train, x_test):
    pre_processor = prep.StandardScaler().fit(x_train)
    x_train = pre_processor.transform(x_train)
    x_test = pre_processor.transform(x_test)
    return x_train, x_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    x_train, x_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1

    auto_encoder = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_data = get_random_block_from_data(x_train, batch_size)
            cost = auto_encoder.partial_fit(batch_data)
            avg_cost += cost[0]/n_samples*batch_size

        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))


    print("total cost: " + str(auto_encoder.calc_total_cost(x_test)))

    print("hidden: " + str(auto_encoder.transform(x_test)) )