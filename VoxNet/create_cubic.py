# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import tensorflow as tf

source_dir = "/home/wzj/Documents/files/dataset/sydney-urban-objects-dataset/objects/"
data_dir = "./xyz/"

names = ['t', 'intensity', 'id',
         'x', 'y', 'z',
         'azimuth', 'range', 'pid']

formats = ['int64', 'uint8', 'uint8',
           'float32', 'float32', 'float32',
           'float32', 'float32', 'int32']
'''
wd bench bicycle biker building bus car cyclist excavator pedestrian pillar pole post scooter
ticket_machine traffic_lights traffic_sign trailer trash tree truck trunk umbrella ute van vegetation
'''

label_dictionary = {'4wd': 0, 'building': 1, 'bus': 2, 'car': 3, 'pedestrian': 4, 'pillar': 5, 'pole': 6,
                    'traffic_lights': 7, 'traffic_sign': 8, 'tree': 9, 'truck': 10, 'trunk': 11, 'ute': 12,
                    'van': 13}
modelNet_label_dictionary = {'bed': 0, 'monitor': 1, 'dresser': 2, 'sofa': 3,
                                 'toilet': 4, 'bathtub': 5, 'chair': 6, 'night_stand': 7, 'desk': 8, 'table': 9}
binType = np.dtype(dict(names=names, formats=formats))
#TODO 增加数据

def get_min_point(data_mat, dim=3):
    '''
    输入一个data_mat，返回沿axis=0方向的最小值；
    '''
    min = np.amin(data_mat, axis=0)
    return min[0:dim]


def get_max_point(data_mat, dim=3):
    '''
    输入一个data_mat，返回沿axis=0方向的最大值；
    '''
    max = np.amax(data_mat, axis=0)
    return max[0:dim]


def load_binary_file(filepath):
    data = np.fromfile(filepath, binType)
    points = np.vstack([data['x'], data['y'], data['z']]).T
    return points


def change_to_xyz_files():
    files = os.listdir(data_dir)
    for file in files:
        if file.endswith("bin"):
            print file
            filename = file.replace("bin", "xyz")
            points = load_binary_file(os.path.join(data_dir, file))
            np.savetxt(os.path.join(data_dir, filename), points, fmt="%3.10f")

    return


# 增加数据
def data_augmentation(points, rotate_angle):
    max_points = get_max_point(points, 3)
    min_points = get_min_point(points, 3)
    center_of_boundingbox = (max_points - min_points) / 2 + min_points
    points[:, 0:3] = points[:, 0:3] - center_of_boundingbox
    np.savetxt("init.xyz", points, fmt="%3.10f")
    xy = points[:, 0:2]
    z = points[:, 2:]
    list_of_rotated_points = []
    num_of_rotated = int(360 / rotate_angle)

    for i in range(num_of_rotated):
        angle = (i * rotate_angle / 180.) * np.pi
        print 'angle:'
        print angle
        rotate_matrix = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        print 'rotated_angles:'
        print rotate_matrix
        temp = np.column_stack((xy.dot(rotate_matrix), z))
        print "rotated points"
        print temp
        list_of_rotated_points.append(temp)

    print list_of_rotated_points

    return data_augmentation


# 使用k邻近算法和pca算法 获得normal信息
def get_features(data_matrix, k_neighbor=3, radius=1):
    nbrs = NearestNeighbors(n_neighbors=k_neighbor, radius=radius, algorithm="ball_tree").fit(data_matrix)
    distance, indices = nbrs.kneighbors(data_matrix, k_neighbor)

    points = data_matrix[indices]
    normal = np.array(map(get_normal_features, points))
    final = np.concatenate((data_matrix, normal), axis=1)

    #print 'features shape: ', final.shape
    #print final
    return final


# 通过pca获得normal信息
def get_normal_features(points):
    pca = PCA()
    pca.fit(points)
    if pca.components_[2][0] > 0:
        normal = pca.components_[2]
    else:
        normal = -pca.components_[2]
    # TODO 这些特征值相互比较的作用
    surfaceness = pca.explained_variance_ratio_[1] - pca.explained_variance_ratio_[2]
    linearness = pca.explained_variance_ratio_[0] - pca.explained_variance_ratio_[1]
    scatterness = pca.explained_variance_ratio_[2]
    features = np.array([surfaceness, linearness, scatterness])
    normal = np.concatenate([normal, features])
    return normal


# 等比例缩放点云
def scale_points(points, scale_list):
    cube_points = np.divide(points[:, 0:3], scale_list)
    points = np.concatenate((cube_points, points[:, 3:]), axis=1)
    return points

#TODO 以中间点为中心向各个方向进行延伸
# 创建包含normal信息的cube体元素 grid_size 应该根据不同的三维图形的大小来变化
def creat_cube_of_object(points, grid_size= 1, x_size=32, y_size=32, z_size=32, feature_dim=3, rotate_angle=0):

    '''
    max_points = get_max_point(points[:, 0:3])
    min_points = get_min_point(points[:, 0:3])

    center_of_boundingbox = (max_points - min_points) / 2
    points[:, 0:3] = points[:, 0:3] - center_of_boundingbox
    '''
    min_points = get_min_point(points[:, 0:3])
    points[:, 0:3] = points[:, 0:3] - min_points
    max_points = get_max_point(points[:, 0:3])
    scatter_list = np.array([1, 1, 1])
    for i in range(3):
        if max_points[i] > x_size:
            scatter_list[i] = max_points[i] / x_size

    corrd_to_cube = scale_points(points, scatter_list)

    #cube = np.zeros((x_size, y_size, z_size, feature_dim))
    # 只放置 0 或 1 不放置 法向量
    cube = np.zeros((x_size, y_size, z_size, feature_dim))

    #print 'cube shape: ', cube.shape
    j = 0
    #print corrd_to_cube.shape

    for element in corrd_to_cube:
        #corrd_to_boundingbox_center = map(lambda x: int(x / grid_size), point[0:3])
        #corrd_to_cube = corrd_to_boundingbox_center + np.array([x_size / 2, y_size / 2, z_size / 2])
        #print corrd_to_cube
        #print corrd_to_cube
        #print element
        condit1 = element[0] >= 0 and element[0] < 32
        condit2 = element[1] >= 0 and element[1] < 32
        condit3 = element[2] >= 0 and element[2] < 32

        if (condit1 and condit2 and condit3):
            cube[int(element[0]), int(element[1]), int(element[2])] = element[3:3 + feature_dim]
            # 不放置法向量 放置 0 或 1
            # cube[int(element[0]), int(element[1]), int(element[2])] = 1
            # print "cube element"
            # print cube[corrd_to_cube[0], corrd_to_cube[1], corrd_to_cube[2]]
        else:
            j = j + 1
            #print "the %dth point do not exist in the cube:" % j
            # print point

    i = 0
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if cube[x][y][z][0] != 0.:
                    i = i+1

    #print 'the sum of numbers which is not zero: %d' % i
    #print 'final cube shape:', cube.shape
    return cube


def parse_file(filename):
    points = np.loadtxt(filename)
    #print 'parse_file: ', points.shape

    return points


def parse_data_to_cube():
    batch_dir = "./batch/"
    data_dir = './xyz/'

    files = os.listdir(batch_dir)

    # 将bin替换成xyz
    '''
    for file in files:
        reader = open(os.path.join(batch_dir, file), "r")
        lines = reader.readlines()
        reader.close()
        writer = open(os.path.join(batch_dir, file), "w")
        for line in lines:
            print line
            line = line.replace('bin', 'xyz')
            print line
            writer.write(line)
        writer.close()
    '''

    labels = []
    idx = 0
    for file in files:
        list_of_data = []
        reader = open(os.path.join(batch_dir, file), "r")
        lines = reader.readlines()
        for line in lines:
            xyz_file = line.replace('\n', '')
            #print 'the result of ', xyz_file
            points = parse_file(os.path.join(data_dir, xyz_file))
            #print "size: ", len(points)
            features = get_features(points)
            cube = creat_cube_of_object(features)
            key = xyz_file.split('.')[0]
            cube = np.reshape(cube, [98304])
            #cube = np.reshape(cube, [32768])
            list_of_data.append(cube)
            labels.append(label_dictionary[key])

        convert_to_records(np.array(list_of_data), labels, idx)
        idx = idx + 1

    return


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to_records(data, labels, idx):

    tf_dir = './tfrecord/'
    tf_file = os.path.join(tf_dir, ('data_batch_%d.tfrecords' % idx))
    print tf_file
    size = data.shape[0]
    print tf_file
    print 'size: ', size
    writer = tf.python_io.TFRecordWriter(tf_file)
    for i in range(size):
        data_raw = list(data[i])
        example = tf.train.Example(features=tf.train.Features(
            feature={"data": _float_feature(data_raw), "label": _int64_feature(int(labels[i]))}))
        writer.write(example.SerializeToString())
    writer.close()
    return

def input_data():

    #tf_file = './tfrecord/temp.tfrecords'
    kinds = modelNet_label_dictionary.keys()
    file_list = ['./tfrecord/train/%s_batch.tfrecords' % kind for kind in kinds]

    filename_queue = tf.train.string_input_producer(file_list, 2)

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
        [data, label], batch_size=4, capacity=2000, min_after_dequeue=1000)
    return image_batch, label_batch


def convert_modelNet_to_records(data, labels, file):

    tf_dir = './tfrecord/'
    # 创建test数据集合
    #tf_file = os.path.join(tf_dir, ('test_%s_batch.tfrecords' % file))
    tf_file = os.path.join(tf_dir, ('%s_batch.tfrecords' % file))

    size = data.shape[0]

    print ' file: %s size: %d ' % (tf_file, size)
    writer = tf.python_io.TFRecordWriter(tf_file)
    for i in range(size):
        data_raw = list(data[i])
        example = tf.train.Example(features=tf.train.Features(
            feature={"data": _float_feature(data_raw), "label": _int64_feature(int(labels[i]))}))
        writer.write(example.SerializeToString())
    writer.close()
    return


#TODO 有NAN异常 待解决
# create tf data used for tensorflow from modelNet
def parse_model_net_to_record():

    modelNet_dataset_dir = '/home/wzj/Documents/files/dataset/ModelNet10/'
    action_type = '/test/'
    tf_records_dir = './tfrecord/'
    files_list = os.listdir(modelNet_dataset_dir)
    print files_list
    for file in files_list:
        is_dir = os.path.isdir(os.path.join(modelNet_dataset_dir, file))
        if is_dir:
            print file
            print modelNet_label_dictionary[file]
            data_files = os.listdir(os.path.join(modelNet_dataset_dir, file+action_type))
            list_cube = []
            labels = []
            print '%s size is %d' % (file, len(data_files))
            step = 1
            for data_file in data_files:

                reader = open(os.path.join(modelNet_dataset_dir+file+action_type, data_file), 'r')
                # 除去OFF文件头
                line = reader.readline()
                line = reader.readline()
                count = int(line.split(' ')[0])
                print "%d step file %s count %d" % (step, data_file, count)
                step = step + 1
                element_list = []
                for i in range(count):
                    # print 'step %d: ' % (i+1)
                    split_line = reader.readline().replace('\n', '').split(' ')
                    temp_list = []
                    for num in split_line:
                        temp_list.append(float(num))
                    # print temp_list
                    element_list.append(temp_list)

                points = np.array(element_list)
                features = get_features(points)
                cube = creat_cube_of_object(features)
                cube = np.reshape(cube, [98304])

                list_cube.append(cube)
                labels.append(modelNet_label_dictionary[file])
                reader.close()

            array_cube = np.array(list_cube)
            array_label = np.array(labels)
            print array_cube.shape
            print array_label.shape
            convert_modelNet_to_records(array_cube, array_label, file)

    return


def input_data_from_modelNet():

    kinds = modelNet_label_dictionary.keys()
    file_list = ['./tfrecord/modelNet/train/%s_batch.tfrecords' % kind for kind in kinds]
    #file_list = ['./tf']
    print file_list

    filename_queue = tf.train.string_input_producer(file_list, 2)

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
        [data, label], batch_size=4, capacity=2000, min_after_dequeue=1000)
    return image_batch, label_batch


def test():
    data, label = input_data_from_modelNet()
    config = tf.ConfigProto()
    # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    for i in range(290):
        i = i+1
        print 'step %d' % i
        x_data, y_label = sess.run([data, label])
        print x_data
        print y_label
        print x_data.shape
        print y_label.shape
        print type(x_data)
        print type(y_label)

    coord.request_stop()
    sess.close()

    return



if __name__ == "__main__":
    #parse_model_net_to_record()
    test()
    #test()
    #data, labels = parse_data_to_cube()
    #convert_to_records(data, labels)
    #convert_to_records(1,2,3)
    #tf.app.run(main=test, argv=[])
    '''
    data, label = read_from_tfile()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while not coord.should_stop():

        print sess.run([data, label])
    '''



    '''
    #change_to_xyz_files()

    data_file = "4wd.0.2299.xyz"
    #points = load_binary_file(filename)
    #print points
    #print len(points)
    #np.savetxt(os.path.join(save_dir, save_file), points, fmt='%3.10f')

    points = np.loadtxt(os.path.join(data_dir, data_file))

    features = get_features(points)
    print features
    cube = creat_cube_of_object(features)
    cube = cube.astype(np.float32)
    print type(cube[0][0][0][0])

    np.savetxt("init.xyz", cube, fmt='%s')

    #get_features(points, k_neighbor=3, radius=1.8)
    #data_augmentation(points,30)
    #print get_min_point(test, 3)
    '''

    '''
    dir_of_dirs = ''
    dirs = os.listdir(dir_of_dirs)
    for dir in dirs:
        files = os.listdir(os.path.join(dir_of_dirs, dir))
        os.chdir(os.path.join(dir_of_dirs, dir))
        os.makedirs('xyz')
        os.makedirs('npy')
        print "处理文件夹%s" % dir
        for file in files:
            print "处理文件%s" % file
            filepath = os.path.join(dir_of_dirs, dir, file)
            points = load_binary_file(filepath)
            np.savetxt(os.path.join(dir_of_dirs, dir, 'xyz', file + '.xyz'), points, fmt='%10.5f')
            np.save(os.path.join(dir_of_dirs, dir, 'npy', file), points)


    dirofDir = 'E:/forest_data/sysdeny'
    dirs = os.listdir(dirofDir)
    list_rate_toSave = []
    for dir in dirs:
        filedir = os.path.join(dirofDir, dir, 'xyz')
        list_of_cube, list_of_occupy_rates = aug_and_normal(filedir, 6)
        np.save(dir, np.array(list_of_cube))
        list_rate_toSave.extend(list_of_occupy_rates)
    np.savetxt('occupy_rate.txt', np.array(list_rate_toSave), fmt='%0.5f')
    '''

