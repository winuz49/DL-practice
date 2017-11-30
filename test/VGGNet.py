import tensorflow as tf
import time



def conv_op(input_op, name, kh, kw, n_out, sh, sw, p):

    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w', shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, [1, sh, sw, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='b')
        activation = tf.nn.relu(tf.nn.bias_add(conv, bias))
        p += [kernel, bias]
        return activation


def fc_op(input_op, name, n_out, p):

    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w', shape=[n_in, n_out],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.0, shape=[n_out]), trainable=True, dtype=tf.float32, name='b')
        activation = tf.nn.relu_layer(input_op, kernel, bias, name=scope)
        p += [kernel, bias]
        return activation


def mpool_op(input_op, name, kh, kw, sh, sw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, sh, sw, 1], padding='SAME', name=name)


def inference_op(input_op, keep_prob):
    p = []

    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, sh=1, sw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, sh=1, sw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, sh=2, sw=2)

    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, sh=1, sw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, sh=1, sw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, sh=2, sw=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, sh=1, sw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, sh=1, sw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, sh=1, sw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, sh=2, sw=2)

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, sh=1, sw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, sh=1, sw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, sh=1, sw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, sh=2, sw=2)

    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, sh=1, sw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, sh=1, sw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, sh=1, sw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, sh=2, sw=2)

    shape = pool5.get_shape()
    flattened_shape = shape[1].value * shape[2].value * shape[3].value
    re_shape = tf.reshape(pool5, [-1, flattened_shape], name='reshape')

    fc6 = fc_op(re_shape, name='fc6', n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc6_drop, keep_prob, name='fc7_drop')

    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    prediction = tf.argmax(softmax, 1)
    return prediction, softmax, fc8, p


def time_tensorflow_run(session, target, feed, info_string):

    num_batches = 100
    total_duration = 0.0

    for i in range(num_batches):
        start = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time()-start

        total_duration += duration

    per_duration  = total_duration/num_batches
    print '%s: %d batches: %.3f total time %.3f per batch' % (info_string, num_batches, total_duration, per_duration)


def run_benchmark():
    batch_size = 10
    with tf.Graph().as_default():
        image_size = 224
        image = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],
                                             dtype=tf.float32, stddev=1e-1))

        keep_prob = tf.placeholder(tf.float32)
        prediction, softmax, fc, p = inference_op(image, keep_prob)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # allocate only as much GPU memory based on runtime allocations
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.Session(config=config)
        session.run(init)

        time_tensorflow_run(session, prediction, {keep_prob: 1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc)
        grad = tf.gradients(objective, p)

        time_tensorflow_run(session, grad, {keep_prob: 0.5}, 'Forward-backward')


if __name__ == '__main__':
    run_benchmark()