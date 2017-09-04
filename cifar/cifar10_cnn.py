# -*- coding:utf-8 -*-
import tensorflow as tf
from cifar import cifar10_input, cifar10
import argparse
import sys
import time
import math
import numpy as np
#max_size = 3000
#batch_size = 128
#data_dir = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data"
FLAGS = None


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)

    return var


def bias_variable(shape, cons):
    return tf.Variable(tf.constant(cons, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def maxPool_3x3_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")


def get_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

    tf.add_to_collection("losses", cross_entropy_mean)
    # get_collection : 获得一个有同样名字变量的list
    # add_n 将list中相同shape的变量相加并返回结果
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


def main(_):

    cifar10.maybe_download_and_extract()
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    image_holder = tf.placeholder(tf.float32, [FLAGS.batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [FLAGS.batch_size])

    print 'get imgaes and labels'
    h_w1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, wl=0.0)
    h_b1 = bias_variable([64], 0.0)
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(image_holder, h_w1), h_b1))
    h_pool1 = maxPool_3x3_2x2(h_conv1)
    h_nor1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    h_w2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, wl=0.0)
    h_b2 = bias_variable([64], 0.1)
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_nor1, h_w2), h_b2))
    h_nor2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    h_pool2 = maxPool_3x3_2x2(h_nor2)

    reshape = tf.reshape(h_pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    h_w3 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
    h_b3 = bias_variable([384], 0.1)
    h_fcn1 = tf.nn.relu(tf.matmul(reshape, h_w3) + h_b3)

    h_w4 = variable_with_weight_loss([384, 192], stddev=0.04, wl=0.004)
    h_b4 = bias_variable([192], 0.1)
    h_fcn2 = tf.nn.relu(tf.matmul(h_fcn1, h_w4) + h_b4)

    h_w5 = variable_with_weight_loss([192, 10], stddev=1/192.0, wl=0.0)
    h_b5 = bias_variable([10], 0.0)
    logits = tf.add(tf.matmul(h_fcn2, h_w5), h_b5)

    loss = get_loss(logits, label_holder)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # in_top_k : 返回输出结果中top k的准确率  top 1 即是得分最高的准确率
    top_k_op = tf.nn.in_top_k(predictions=logits, targets=label_holder, k=1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # start_queue_runners 启动完成前面collection任务的线程
    tf.train.start_queue_runners()

    begin = time.time()
    for step in range(FLAGS.max_size):
        start_time = time.time()
        images_batch, labels_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_step, loss], feed_dict={
            image_holder: images_batch, label_holder: labels_batch})

        duration = time.time() - start_time
        if step % 10 == 0:
            example_per_sec = duration / FLAGS.batch_size
            sec_per_batch = float(duration)

            format_str =("step %d loss=%.2f (%.1f examples/sec; %.3f sec/batch)")
            print(format_str % (step, loss_value, example_per_sec, sec_per_batch))

    total_duration = time.time() - begin
    print("total train time %d sec" % total_duration)

    num_examples = 10000
    num_iter = int(math.ceil(num_examples/FLAGS.batch_size))
    total_count = num_iter*FLAGS.batch_size
    true_count = 0

    for step in range(num_iter):
        images_batch, labels_batch = sess.run([images_test, labels_test])
        prediction = sess.run([top_k_op],
                              feed_dict={image_holder: images_batch, label_holder: labels_batch})
        true_count += np.sum(prediction)

    precision = float(true_count)/float(total_count)
    print ("total count: %d true count: %d" % (total_count, true_count))
    print("precision %.3f " % precision)

if __name__ == "__main__":
    print "cifar example"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=
    "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/cifar-10-batches-bin", help="Directory for storing input data")
    parser.add_argument("--batch_size", type=int, default=256, help="the number of examples each batch to train")
    parser.add_argument("--max_size", type=int, default=6000, help="the max number of examples need to train")
    FLAGS = parser.parse_args()
    print FLAGS
    tf.app.run(main=main, argv=[sys.argv[0]])

    print 'hh'