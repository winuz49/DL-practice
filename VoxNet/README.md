# VoxNet

### 数据处理过程
sydney 数据集处理流程：
（1）首先将bin文件转换成xyz文件
（2）然后将xyz文件转换成cube元素
（3）将批量的cube元素转换成tensorflow支持的TF文件

modelNet（10个类别） 数据集处理流程：
(1) 分别将每个类别下的每个off文件转成cube
(2) 将整个类别目录下的文件存储成TF格式文件

### 训练测试
voxnet读取TF文件进行训练预测




