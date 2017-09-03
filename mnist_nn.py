# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import argparse
FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bia_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def deep_nn(x):
    """deepnn for mnist digit classification
    :param x: tensor shape:[None,784]
    :return:  A tuple (y, keep_prob). y tensor shape :[10] the prediction result .
     keep_prob the rate of dropout
    """
    with tf.name_scope("reshape"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])


    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bia_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)

    with tf.name_scope("pool1"):
        h_pool1= max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bia_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("fcn1"):
        W_fcn1 = weight_variable([7*7*64, 1024])
        b_fcn1 = bia_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fcn1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcn1)+b_fcn1)

    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        h_fcn1_drop = tf.nn.dropout(h_fcn1, keep_prob)

    with tf.name_scope("fcn2"):
        W_fcn2 = weight_variable([1024, 10])
        b_fcn2 = bia_variable([10])

        y_conv = tf.matmul(h_fcn1_drop, W_fcn2)+b_fcn2

    return y_conv, keep_prob


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

    y_conv, keep_prob = deep_nn(x)

    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope("adam_optimizer"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # eval 执行该方法 如accuracy.eval 则运行graph中accuracy前所有操作 直到获得accuracy的值
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("%d step training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("testing accuracy %g" % (accuracy.eval(feed_dict=
                                                 {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="MNIST_data", help="Directory for storing input data")
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
