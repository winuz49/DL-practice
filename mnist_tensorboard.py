# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import argparse
max_step = 1000
learning_rate = 1e-4
dropout = 0.9
data_dir = "MNIST_data"
log_dir = "/home/wzj/Documents/tensorflow_env/log"
mnist = input_data.read_data_sets(data_dir, one_hot=True)

sess = tf.InteractiveSession()

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bia_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def variable_summary(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)


def nn_layer(input_tensor, input_dim, output_dim,layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weight_variable([input_dim,output_dim])
            variable_summary(weights)
        with tf.name_scope("bias"):
            biases = bia_variable([output_dim])
            variable_summary(biases)
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weights)+biases
            tf.summary.histogram("pre_activate", preactivate)
        activations = act(preactivate, name="activation")
        tf.summary.histogram("activations", activations)
        return activations


def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0

    return {x: xs, y_:ys, keep_prob: k}


def main(_):

    with tf.name_scope("reshape"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image("input", x_image, 10)
    hidden1 = nn_layer(x, 784,500,"layer1")
    with tf.name_scope("dropout"):
        tf.summary.scalar("dropout_keep_prob", keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)
    y = nn_layer(dropped, 500, 10, "layer2", act=tf.identity)
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.name_scope("correction_prediction"):
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir+"/test", sess.graph)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    for i in range(max_step):
        if i % 10 == 0:
            summary,ace = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary)
            print(" Accuracy at step %s : %s" % (i, ace))
        if i % 100 == 99:
            run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ =sess.run([merged, train_step], feed_dict=feed_dict(True),
                                 options=run_option,run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step%03d" % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+"/model.ckpt", i)
            print("adding run metadata for", i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="MNIST_data", help="Directory for storing input data")
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])


