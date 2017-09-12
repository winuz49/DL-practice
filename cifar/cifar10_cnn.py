# -*- coding:utf-8 -*-
import tensorflow as tf
import argparse
import sys
import time
import math
import numpy as np
import pickle
import os
FLAGS = None
max_size = 1000
data_size = 50000
model_path = "./CIFAR_data/"


def unpickle(file):
    with open(file, "rb") as content:
        dict = pickle.load(content)
    return dict


def test_cifar():
    train_folder = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/cifar-10-batches-py/data_batch_"
    train_file = train_folder + str(1)
    dict = unpickle(train_file)
    print dict["data"].shape
    print len(dict["labels"])


def normalization_array(array):
    arr_sum = np.sum(array, axis=0)
    x_mean = arr_sum / np.float32(array.shape[0] * 255)
    arr_result = array / np.float32(255) - x_mean
    return arr_result


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_records():
    filename = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/cifar-10-batches-py/data_batch_"
    tf_records = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/tfrecord/data_batch_"

    '''
    print "compare to read"
    for i in range(10):
        print x_data[i*2:(i+1)*2, :]
        print y_data[i*2:(i+1)*2]
    print 'end'
    '''
    print "train:"
    for i in range(1, 6):
        dict = unpickle(filename+str(i))
        x_data = dict["data"]
        y_data = np.array(dict["labels"])
        size = x_data.shape[0]
        writer = tf.python_io.TFRecordWriter(tf_records+str(i)+".tfrecords")
        for i in range(size):
            data_raw = x_data[i].tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={"data": _byte_feature(data_raw), "label": _int64_feature(int(y_data[i]))}))
            writer.write(example.SerializeToString())
        writer.close()

    print "test:"
    test_file = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/cifar-10-batches-py/test_batch"
    test_records = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/tfrecord/test_batch"

    dict = unpickle(test_file)
    x_data = dict["data"]
    y_data = np.array(dict["labels"])
    size = x_data.shape[0]
    writer = tf.python_io.TFRecordWriter(test_records + ".tfrecords")
    for i in range(size):
        data_raw = x_data[i].tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={"data": _byte_feature(data_raw), "label": _int64_feature(int(y_data[i]))}))
        writer.write(example.SerializeToString())
    writer.close()

    return


def _parse_function(example_proto):
    features ={"data": tf.FixedLenFeature(
        (), tf.string, default_value=""), "label": tf.FixedLenFeature([], tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["data"], parsed_features["label"]


def read_from_record():
    test_file = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/tfrecord/test_batch.tfrecords"
    i = 0
    for serialized_example in tf.python_io.tf_record_iterator(test_file):
        i = i+1
        if i == 10:
            break
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        data = example.features.feature["data"].bytes_list.value
        label = example.features.feature["label"].int64_list.value[0]
        print data
        print label

    return


def test_main():

    image_batch, label_batch = input_data(True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    i = 1

    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            print "step %d: " % i
            x_train, y_train = sess.run([image_batch, label_batch])
            print x_train.shape
            print y_train.shape
            print x_train.dtype
            print y_train.dtype
            x_train = tf.cast(tf.decode_raw(x_train, tf.uint8), tf.float32) / 255
            y_train = tf.cast(y_train, tf.int32)
            print sess.run([x_train, y_train])
            i = i+1
    except tf.errors.OutOfRangeError:
        print "Done training -- epoch limit reached"
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    sess = tf.Session(config=config)
    image_batch, label_batch = input_data(True, batch_size=128, num_epochs=1)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            print "step %d: " % i
            x_train, y_train = sess.run([image_batch, label_batch])
            x_train = tf.cast(tf.decode_raw(x_train, tf.uint8), tf.float32) / 255
            y_train = tf.cast(y_train, tf.int32)
            print sess.run([x_train, y_train])
            i = i+1
    except tf.errors.OutOfRangeError:
        print "Done testing -- epoch limit reached"
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    return


def read_from_record_from_dataset():
    train_dir = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/tfrecord/"
    test_file = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/tfrecord/test_batch.tfrecords"
    train_files = [os.path.join(train_dir, "data_batch_%d.tfrecords") % i for i in range(1, 6)]
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(2)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    config = tf.ConfigProto()
    # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)

    sess.run(iterator.initializer, feed_dict={filenames: train_files})

    for i in range(5):
        print "step %d " % i

        data, label = sess.run(next_element)
        data = tf.decode_raw(data, tf.uint8)
        label = tf.cast(label, tf.int32)
        re_data = tf.reshape(data, [2, 3072])
        nor_data = tf.cast(re_data, tf.float32) * 1.0/255 - 0.5
        print sess.run(nor_data)
        print sess.run(label)

    return


def test_tf_record():
    filename = "/home/wzj/PycharmProjects/DL-practice/CIFAR_data/cifar-10-batches-py/test_batch"
    dict = unpickle(filename)
    x_data = tf.cast(dict["data"][0:100, :], tf.float32)
    y_data = tf.cast(np.array(dict["labels"][0:100]), tf.int32)
    print x_data.shape
    print y_data.shape
    #     images, labels = tf.train.shuffle_batch 根据源数据集和batch大小获得batch数据集
    images, labels = tf.train.shuffle_batch(
        [x_data[0,:], y_data[0]], batch_size=2, enqueue_many=False, capacity=5000, min_after_dequeue=1000)

    op = images+1

    sess = tf.InteractiveSession()
    print 'hhh'
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    print 'hhh'
    print sess.run(op)
    print images.shape
    print labels.shape
    print 'hhh'


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
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")


def get_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

    tf.add_to_collection("losses", cross_entropy_mean)
    # get_collection : 获得一个有同样名字变量的list
    # add_n 将list中相同shape的变量相加并返回结果
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


def input_data(eval_true=False, batch_size=128, num_epochs=1):
    test_file = "./CIFAR_data/tfrecord/test_batch.tfrecords"
    train_folder ="./CIFAR_data/tfrecord/"
    train_files = [train_folder+"data_batch_%d.tfrecords" % i for i in range(1, 6)]
    print train_files
    if not eval_true:
        filename_queue = tf.train.string_input_producer(train_files, num_epochs=num_epochs)
    else:
        filename_queue = tf.train.string_input_producer([test_file], num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        "data": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })
    label = features["label"]
    data = features["data"]
    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    return image_batch, label_batch


def inference(image_batch):

    print 'get images and labels'
    h_w1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, wl=0.0)
    h_b1 = bias_variable([64], 0.0)
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(image_batch, h_w1), h_b1))
    h_pool1 = maxPool_3x3_2x2(h_conv1)
    p_shape1 = h_pool1.get_shape()
    print "shape1 : ", p_shape1
    h_nor1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    h_w2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, wl=0.0)
    h_b2 = bias_variable([64], 0.1)
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_nor1, h_w2), h_b2))
    h_nor2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    h_pool2 = maxPool_3x3_2x2(h_nor2)
    p_shape2 = h_pool2.get_shape()
    print "shape2 : ", p_shape2

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

    return logits


def train(_):

    image_batch, label_batch = input_data(eval_true=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch)
    with tf.name_scope("reshape") as scope:
        image_batch = tf.cast(tf.decode_raw(image_batch, tf.uint8), tf.float32) / 255
        label_batch = tf.cast(label_batch, tf.int32)
        image_batch = tf.transpose(tf.reshape(image_batch, [FLAGS.batch_size, 3, 32, 32]), [0, 2, 3, 1])
        label_batch = tf.reshape(label_batch, [FLAGS.batch_size])

    logits = inference(image_batch)
    loss = get_loss(logits, label_batch)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # in_top_k : 返回输出结果中top k的准确率  top 1 即是得分最高的准确率
    top_k_op = tf.nn.in_top_k(predictions=logits, targets=label_batch, k=1)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(init_op)
    saver = tf.train.Saver()

    begin = time.time()
    try:
        coord = tf.train.Coordinator()
        # start_queue_runners 启动完成前面collection任务的线程
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
        save_path = saver.save(sess,model_path+"model.ckpt")
        coord.request_stop()
        print "model save path:", save_path

    coord.join(threads)
    sess.close()
    total_duration = time.time() - begin
    print("total train time %d sec" % total_duration)


def test(_):
    image_batch, label_batch = input_data(eval_true=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch)

    with tf.name_scope("reshape") as scope:
        image_batch = tf.cast(tf.decode_raw(image_batch, tf.uint8), tf.float32) / 255
        label_batch = tf.cast(label_batch, tf.int32)
        image_batch = tf.transpose(tf.reshape(image_batch, [FLAGS.batch_size, 3, 32, 32]), [0, 2, 3, 1])
        label_batch = tf.reshape(label_batch, [FLAGS.batch_size])

    logits = inference(image_batch)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(init_op)
    coord = tf.train.Coordinator()
    # start_queue_runners 启动完成前面collection任务的线程
    saver = tf.train.Saver()
    saver.restore(sess, model_path+"model.ckpt")

    begin = time.time()

    num_examples = 10000
    total_count = num_examples*FLAGS.epoch
    true_count = 0
    test_loss = get_loss(logits, label_batch)
    # in_top_k : 返回输出结果中top k的准确率  top 1 即是得分最高的准确率
    test_top_k_op = tf.nn.in_top_k(predictions=logits, targets=label_batch, k=1)
    step = 1
    try:
        coord = tf.train.Coordinator()
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

    print "final step:",step
    coord.join(threads)
    total_duration = time.time() - begin
    print("total train time %d sec" % total_duration)
    precision = float(true_count)/float(total_count)
    total_duration = time.time() - begin
    print("total test time %d sec" % total_duration)
    print ("total count: %d true count: %d" % (total_count, true_count))
    print("precision %.3f " % precision)
    sess.close()


if __name__ == "__main__":

    print "cifar example"
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="the number of examples each batch to train")
    parser.add_argument("--epoch", type=int, default=16, help="the max number of examples need to train")
    parser.add_argument("--action", type=str, default="train", help="the action you want to do ")

    FLAGS = parser.parse_args()
    print FLAGS
    if FLAGS.action == "train":
        func = train
    elif FLAGS.action == "test":
        func = test
    tf.app.run(main=func, argv=[sys.argv[0]])

