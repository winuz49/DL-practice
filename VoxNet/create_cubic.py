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

label_dictionary = {'4wd': 0, 'bench': 1, 'bicycle': 2, 'biker': 3, 'building': 4,'bus': 5, 'car': 6,
                    'cyclist':7, 'excavator':8, 'pedestrian':9, 'pillar':10, 'pole':11, 'post':12,
                    'scooter':13, 'ticket_machine':14, 'traffic_lights':15, 'traffic_sign':16, 'trailer':17,
                    'trash':18, 'tree':19, 'truck':20, 'trunk':21, 'umbrella':22, 'ute':23, 'van':24, 'vegetation':25}
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
    file_list = ['./tfrecord/data_batch_%d.tfrecords' % i for i in range(4)]

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
    data, label = input_data()
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


def create_dataset():
    file_list_dir = './batch'
    list_files = os.listdir(file_list_dir)
    print list_files
    '''
    # replace bin by xyz
    for list_file in list_files:
        reader = open(os.path.join(file_list_dir, list_file), 'r')
        lines = reader.readlines()
        reader.close()
        writer = open(os.path.join(file_list_dir, list_file), 'w')
        for line in lines:
            line = line.replace('bin', 'xyz')
            print line
            writer.write(line)
        print lines
        writer.close()
    '''
    idx = 0
    for list_file in list_files:
        reader = open(os.path.join(file_list_dir, list_file), 'r')



if __name__ == "__main__":
    test()
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

