# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import argparse
FLAGS = None


def _activation_summary(x):
    """Helper to create summaries for activations."""
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + "/activation", x)
    tf.summary.scalar(tensor_name + "/sparsity", tf.nn.zero_fraction(x))


def print_activation(t):
    print(t.op.name, " ", t.get_shape().as_list())


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)

    return var


def bias_variable(shape, cons):
    return tf.Variable(tf.constant(cons, shape=shape), dtype=tf.float32)


def conv_2d(x, W, stride, padding="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def inference(boxes):

    with tf.name_scope("conv1") as scope:
        kernel = variable_with_weight_loss([5, 5, 5, 3, 64], stddev=0.1, wl=0.0)
        bias = bias_variable([64], 0.0)
        conv1 = tf.nn.conv3d(boxes, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bias), name=scope)
        _activation_summary(conv1)
        print_activation(conv1)

    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name="pool1")
    print_activation(pool1)

    with tf.name_scope("conv2") as scope:
        kernel = variable_with_weight_loss([5, 5, 5, 64, 128], stddev=0.1, wl=0.0)
        bias = bias_variable([128], 0.0)
        conv2 = tf.nn.conv3d(pool1, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bias), name=scope)
        _activation_summary(conv2)
        print_activation(conv2)
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name="pool2")
    print_activation(pool2)

    with tf.name_scope("conv3") as scope:
        kernel = variable_with_weight_loss([5, 5, 5, 128, 128], stddev=0.1, wl=0.0)
        bias = bias_variable([128], 0.1)
        conv3 = tf.nn.conv3d(pool2, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv3, bias), name=scope)
        _activation_summary(conv3)
        print_activation(conv3)
    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name="pool3")
    print_activation(pool3)

    with tf.name_scope("fcn1") as scope:
        reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        print ("dim: ", dim)
        kernel = variable_with_weight_loss([dim, 128], stddev=0.04, wl=0.004)
        bias = bias_variable([128], 0.1)
        fcn1 = tf.nn.relu(tf.matmul(reshape, kernel) + bias, name=scope)
        _activation_summary(fcn1)
        print_activation(fcn1)

    with tf.name_scope("fcn2") as scope:
        kernel = variable_with_weight_loss([128, 10], stddev=0.04, wl=0.004)
        bias = bias_variable([10], 0.1)
        fcn2 = tf.nn.relu(tf.matmul(fcn1, kernel) + bias, name=scope)
        _activation_summary(fcn2)
        print_activation(fcn2)

    with tf.name_scope("softmax_linear") as scope:
        kernel = variable_with_weight_loss([10, FLAGS.num_classes], stddev=0.0001, wl=0.0)
        bias = bias_variable([FLAGS.num_classes], 0.0)
        softmax_linear = tf.nn.relu(tf.matmul(fcn2, kernel) + bias, name=scope)
        _activation_summary(softmax_linear)
        print_activation(softmax_linear)

    return softmax_linear


def get_loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from dataset
    Returns:
      Loss tensor of type float.
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection(name="losses", value=cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


def test(_):
    print 'test'
    with tf.Graph().as_default():
        image_size = 32
        images = tf.Variable(tf.random_normal(
            [FLAGS.batch_size, image_size, image_size, image_size, 3], dtype=tf.float32, stddev=0.1))

        print_activation(images)
        logits = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./imagenet_data/train", help="Directory for input data")
    parser.add_argument("--batch_size", type=int, default=32, help="the number of examples each batch to train")
    parser.add_argument("--max_size", type=int, default=3000, help="the max number of examples need to train")
    parser.add_argument("--num_classes", type=int, default=10, help="the  classes of the examples ")
    FLAGS = parser.parse_args()
    tf.app.run(main=test, argv=[sys.argv[0]])

