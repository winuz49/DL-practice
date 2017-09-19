# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import argparse
import numpy as np
FLAGS = None
import time
model_path = "./checkpoints/"
modelNet_label_dictionary = {'bed': 0, 'monitor': 1, 'dresser': 2, 'sofa': 3,
                                 'toilet': 4, 'bathtub': 5, 'chair': 6, 'night_stand': 7, 'desk': 8, 'table': 9}
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
        kernel = variable_with_weight_loss([dim, 1024], stddev=0.04, wl=0.004)
        bias = bias_variable([1024], 0.1)
        fcn1 = tf.nn.relu(tf.matmul(reshape, kernel) + bias, name=scope)
        _activation_summary(fcn1)
        print_activation(fcn1)

    with tf.name_scope("fcn2") as scope:
        kernel = variable_with_weight_loss([1024, 512], stddev=0.04, wl=0.004)
        bias = bias_variable([512], 0.1)
        fcn2 = tf.nn.relu(tf.matmul(fcn1, kernel) + bias, name=scope)
        _activation_summary(fcn2)
        print_activation(fcn2)

    with tf.name_scope("softmax_linear") as scope:
        kernel = variable_with_weight_loss([512, FLAGS.num_classes], stddev=0.0001, wl=0.0)
        bias = bias_variable([FLAGS.num_classes], 0.0)
        softmax_linear = tf.nn.relu(tf.matmul(fcn2, kernel) + bias, name=scope)
        _activation_summary(softmax_linear)
        print_activation(softmax_linear)

    return softmax_linear


def input_data(eval_true=False):
    if not eval_true:
        #file_list = ['./tfrecord/data_batch_%d.tfrecords' % i for i in range(3)]
        file_list = ['./tfrecord/data_batch_1.tfrecords', './tfrecord/data_batch_2.tfrecords',
                     './tfrecord/data_batch_3.tfrecords']
        filename_queue = tf.train.string_input_producer(file_list, FLAGS.epoch)
    else:
        file_list = ['./tfrecord/data_batch_0.tfrecords']
        filename_queue = tf.train.string_input_producer(file_list, 1)

    #print filename_queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    #print serialized_example
    # normal setting "data": tf.FixedLenFeature([98304], tf.float32)

    features = tf.parse_single_example(serialized_example, features={
        "data": tf.FixedLenFeature([98304], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64)
    })
    data = features["data"]
    label = features["label"]
    print data, label

    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=FLAGS.batch_size, capacity=2000, min_after_dequeue=1000)
    return image_batch, label_batch


def input_data_from_modelNet(eval_true=False):

    kinds = modelNet_label_dictionary.keys()
    if not eval_true:
        file_list = ['./tfrecord/modelNet/train/%s_batch.tfrecords' % kind for kind in kinds]
        filename_queue = tf.train.string_input_producer(file_list, FLAGS.epoch)
    else:
        file_list = ['./tfrecord/modelNet/test/test_%s_batch.tfrecords' % kind for kind in kinds]
        filename_queue = tf.train.string_input_producer(file_list, 1)

    print filename_queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    print serialized_example

    features = tf.parse_single_example(serialized_example, features={
        "data": tf.FixedLenFeature([98304], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64)
    })
    data = features["data"]
    label = features["label"]
    print data, label

    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=FLAGS.batch_size, capacity=2000, min_after_dequeue=1000)
    return image_batch, label_batch


def train(_):
    cube_batch, label_batch = input_data()

    with tf.name_scope("reshape") as scope:
        cube_batch = tf.cast(cube_batch, tf.float32)
        label_batch = tf.cast(label_batch, tf.int32)
        cube_batch = tf.reshape(cube_batch, [FLAGS.batch_size, 32, 32, 32, 3])
        label_batch = tf.reshape(label_batch, [FLAGS.batch_size])
        print 'cube', cube_batch
        print 'label', label_batch


    logits = inference(cube_batch)
    loss = get_loss(logits, label_batch)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    config = tf.ConfigProto()
    # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()

    begin = time.time()
    try:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 1
        while not coord.should_stop():
            start_time = time.time()

            _, loss_value = sess.run([train_step, loss])

            duration = time.time() - start_time
            if step % 10 == 0:
                example_per_sec = duration / FLAGS.batch_size
                sec_per_batch = float(duration)
                format_str = ("step %d loss=%.2f (%.1f examples/sec; %.3f sec/batch)")
                print(format_str % (step, loss_value, example_per_sec, sec_per_batch))
            step = step + 1
    except tf.errors.OutOfRangeError:
        print "Done train -- epoch limit reached"
    finally:
        save_path = saver.save(sess, model_path + "model.ckpt")
        print "model save path:", save_path
        coord.request_stop()

    coord.join(threads)
    total_duration = time.time() - begin
    print("total train time %d sec" % total_duration)
    sess.close()


def test(_):

    cube_batch, label_batch = input_data(eval_true=True)

    with tf.name_scope("reshape") as scope:
        cube_batch = tf.cast(cube_batch, tf.float32)
        label_batch = tf.cast(label_batch, tf.int32)
        cube_batch = tf.reshape(cube_batch, [FLAGS.batch_size, 32, 32, 32, 3])
        label_batch = tf.reshape(label_batch, [FLAGS.batch_size])
        print 'cube', cube_batch
        print 'label', label_batch

    logits = inference(cube_batch)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    saver.restore(sess, model_path+"model.ckpt")

    begin = time.time()

    total_count = int(FLAGS.test_total)
    true_count = 0
    test_loss = get_loss(logits, label_batch)
    # in_top_k : 返回输出结果中top k的准确率  top 1 即是得分最高的准确率
    test_top_k_op = tf.nn.in_top_k(predictions=logits, targets=label_batch, k=1)
    step = 1
    try:


        # start_queue_runners 启动完成前面collection任务的线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while not coord.should_stop():
            start_time = time.time()

            prediction, loss_value = sess.run([test_top_k_op, test_loss])
            true_count += np.sum(prediction)

            duration = time.time() - start_time
            if step % 10 == 0:
                example_per_sec = duration / FLAGS.batch_size
                sec_per_batch = float(duration)
                format_str = ("step %d loss=%.2f (%.1f examples/sec; %.3f sec/batch)")
                print(format_str % (step, loss_value, example_per_sec, sec_per_batch))
            step = step + 1
    except tf.errors.OutOfRangeError:
        print "Done test -- epoch limit reached"
    finally:
        coord.request_stop()

    print "final step:", step
    coord.join(threads)
    precision = float(true_count)/float(total_count)
    total_duration = time.time() - begin
    print("total test time %d sec" % total_duration)
    print ("total count: %d true count: %d" % (total_count, true_count))
    print("precision %.3f " % precision)
    sess.close()


def get_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

    tf.add_to_collection("losses", cross_entropy_mean)
    # get_collection : 获得一个有同样名字变量的list
    # add_n 将list中相同shape的变量相加并返回结果
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="the number of examples each batch to train")
    parser.add_argument("--epoch", type=int, default=20, help="the max number of examples need to train")
    parser.add_argument("--num_classes", type=int, default=10, help="the  classes of the examples ")
    parser.add_argument("--action", type=str, default="train", help="the action you want to do ")
    parser.add_argument("--test_total", type=str, default=1000, help="the total count of test examples ")
    FLAGS = parser.parse_args()
    print FLAGS
    if FLAGS.action == "train":
        func = train
    elif FLAGS.action == "test":
        func = test
    tf.app.run(main=func, argv=[sys.argv[0]])




