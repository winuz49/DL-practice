# -*- coding:utf-8 -*-
import tensorflow as tf
import sys
import argparse
import pickle
import time
FLAGS = None


def unpickle(file):
    with open(file, "rb") as content:
        dict = pickle.load(content)
    return dict


def read_data_set():
    val_dir = "./train/val_data"
    dict = unpickle(val_dir)

    x = dict["data"]
    y = dict["labels"]
    image_size = x.shape[0]

    print "shape of x", x.shape

    valuequeue = tf.train.input_producer(x, shuffle=False)
    labelqueue = tf.train.input_producer(y, shuffle=False)

    images = valuequeue.dequeue()
    y_train = labelqueue.dequeue()

    print images.get_shape()
    x_train = tf.cast(images, tf.float32)
    y_train = tf.cast(y_train, tf.int32)

    x_train = tf.reshape(x_train, ([1, 3, 16, 16]))
    x_train = tf.transpose(x_train, [0, 2, 3, 1])

    print_activation(x_train)
    return x_train, y_train


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


def inference(images):
    with tf.name_scope("conv1") as scope:
        kernel = variable_with_weight_loss([11, 11, 3, 64], stddev=0.1, wl=0.0)
        bias = bias_variable([64], 0.0)
        conv1 = tf.nn.relu(tf.nn.bias_add(conv_2d(images, kernel, stride=4), bias), name=scope)
        _activation_summary(conv1)
        print_activation(conv1)
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn1")
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")
    print_activation(pool1)

    with tf.name_scope("conv2") as scope:
        kernel = variable_with_weight_loss([5, 5, 64, 192], stddev=0.1, wl=0.0)
        bias = bias_variable([192], 0.0)
        conv2 = tf.nn.relu(tf.nn.bias_add(conv_2d(pool1, kernel, stride=1), bias), name=scope)
        _activation_summary(conv2)
        print_activation(conv2)
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn2")
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")
    print_activation(pool2)

    with tf.name_scope("conv3") as scope:
        kernel = variable_with_weight_loss([3, 3, 192, 384], stddev=0.1, wl=0.0)
        bias = bias_variable([384], 0.0)
        conv3 = tf.nn.relu(tf.nn.bias_add(conv_2d(pool2, kernel, stride=1), bias), name=scope)
        _activation_summary(conv3)
        print_activation(conv3)


    with tf.name_scope("conv4") as scope:
        kernel = variable_with_weight_loss([3, 3, 384, 256], stddev=0.1, wl=0.0)
        bias = bias_variable([256], 0.0)
        conv4 = tf.nn.relu(tf.nn.bias_add(conv_2d(conv3, kernel, stride=1), bias), name=scope)
        _activation_summary(conv4)
        print_activation(conv4)

    with tf.name_scope("conv5") as scope:
        kernel = variable_with_weight_loss([3, 3, 256, 256], stddev=0.1, wl=0.0)
        bias = bias_variable([256], 0.1)
        conv5 = tf.nn.relu(tf.nn.bias_add(conv_2d(conv4, kernel, stride=1), bias), name=scope)
        _activation_summary(conv5)
        print_activation(conv5)
    lrn5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn5")
    pool5 = tf.nn.max_pool(lrn5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool5")
    print_activation(pool5)

    with tf.name_scope("fcn1") as scope:
        reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        print ("dim: ", dim)
        kernel = variable_with_weight_loss([dim, 4096], stddev=0.04, wl=0.004)
        bias = bias_variable([4096], 0.1)
        fcn1 = tf.nn.relu(tf.matmul(reshape, kernel) + bias, name=scope)
        _activation_summary(fcn1)
        print_activation(fcn1)

    with tf.name_scope("fcn2") as scope:
        kernel = variable_with_weight_loss([4096, 4096], stddev=0.04, wl=0.004)
        bias = bias_variable([4096], 0.1)
        fcn2 = tf.nn.relu(tf.matmul(fcn1, kernel) + bias, name=scope)
        _activation_summary(fcn2)
        print_activation(fcn2)

    with tf.name_scope("fcn3") as scope:
        kernel = variable_with_weight_loss([4096, 1000], stddev=0.04, wl=0.004)
        bias = bias_variable([1000], 0.1)
        fcn3 = tf.nn.relu(tf.matmul(fcn2, kernel) + bias, name=scope)
        _activation_summary(fcn3)
        print_activation(fcn3)

    with tf.name_scope("softmax_linear") as scope:
        kernel = variable_with_weight_loss([1000, FLAGS.num_classes], stddev=0.0001, wl=0.0)
        bias = bias_variable([FLAGS.num_classes], 0.0)
        softmax_linear = tf.nn.relu(tf.matmul(fcn3, kernel) + bias, name=scope)
        _activation_summary(softmax_linear)
        print_activation(softmax_linear)

    return softmax_linear


def get_loss(loss, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from dataset
    Returns:
      Loss tensor of type float.
    """
    #labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=loss, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection(name="losses", value=cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


def inputdata(data_dir):

    return


def test(_):

    with tf.Graph().as_default():
        #image_size = 224
        #images = tf.Variable(tf.random_normal(
            #[FLAGS.batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=0.1))

        #print_activation(images)
        #logits = inference(images)
        x_train, y_train = read_data_set()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        result = tf.Print(y_train, [y_train])
        result = result + 1
        print sess.run([result])
        print 'jjj'

"""
def train():


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
q!
        duration = time.time() - start_time
        if step % 10 == 0:
            example_per_sec = duration / FLAGS.batch_size
            sec_per_batch = float(duration)
            format_str = ("step %d loss=%.2f (%.1f examples/sec; %.3f sec/batch)")
            print(format_str % (step, loss_value, example_per_sec, sec_per_batch))

    total_duration = time.time() - begin
    print("total train time %d sec" % total_duration)

    return
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=
    "/home/wzj/PycharmProjects/DL-practice/imagenet_data/train", help="Directory for storing input data")
    parser.add_argument("--batch_size", type=int, default=128, help="the number of examples each batch to train")
    parser.add_argument("--max_size", type=int, default=3000, help="the max number of examples need to train")
    parser.add_argument("--num_classes", type=int, default=10, help="the  classes of the examples ")
    FLAGS = parser.parse_args()
    tf.app.run(main=test, argv=[sys.argv[0]])
