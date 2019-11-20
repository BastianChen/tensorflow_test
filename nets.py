import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class MainNet:
    def __init__(self):
        # 定义输入数据的大小
        self.data = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        # 定义标签数据的大小
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.train = tf.Variable(tf.constant(False))

        # 定义网络层参数
        # self.conv1 = tf.nn.conv2d(self.data, filter=[3, 3, 1, 128], strides=[1, 1, 1, 1], padding="SAME")
        # self.conv1 = conv2d([None, 28, 28, 1], weight_variable([3, 3, 1, 128]))  # n,128,26,26
        # # 定义池化层
        # self.pool1 = max_pool_2x2([None, 128, 26, 26])  # n,128,13,13
        # self.conv2 = conv2d([None, 128, 13, 13], [3, 3, 1, 256])  # n,256,11,11
        # self.pool2 = max_pool_2x2([None, 256, 11, 11])  # n,256,5,5
        # self.conv3 = conv2d([None, 256, 5, 5], [3, 3, 1, 512])  # n,512,3,3
        self.linear1 = weight_variable([7 * 7 * 256, 1024])
        self.linear_bias1 = bias_variable([1024])
        self.linear2 = weight_variable([1024, 10])
        self.linear_bias2 = bias_variable([10])

        self.forward()
        self.backward()

    def forward(self):
        # input = tf.reshape(self.data, [-1, 28, 28, 1])
        self.conv1 = conv2d(self.data, weight_variable([5, 5, 1, 128]))  # n,128,26,26
        # 定义池化层
        self.pool1 = max_pool_2x2(self.conv1)  # n,128,13,13
        self.conv2 = conv2d(self.pool1, weight_variable([5, 5, 128, 256]))  # n,256,11,11
        self.pool2 = max_pool_2x2(self.conv2)  # n,256,5,5
        self.pool2 = tf.reshape(self.pool2, [-1, 7 * 7 * 256])
        self.linear_layer1 = tf.nn.relu(tf.matmul(self.pool2, self.linear1) + self.linear_bias1)
        self.linear_layer2 = tf.nn.softmax(tf.matmul(self.linear_layer1, self.linear2) + self.linear_bias2)

    def backward(self):
        self.loss = tf.reduce_mean((self.linear_layer2 - self.label) ** 2)
        self.opt = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)


# 权重参数初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断的正态分布，标准差stddev
    return tf.Variable(initial)


# 初始化bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def conv2d(input, filter, train=False):
    # stride的四个参数：[batch, height, width, channels], [batch_size, image_rows, image_cols, number_of_colors]
    # height, width就是图像的高度和宽度，batch和channels在卷积层中通常设为1
    # 做完归一化再激活
    b_conv = bias_variable([filter.shape[-1]])
    data = tf.nn.relu(
        batch_norm_layer((tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME') + b_conv), train))
    return data


# 定义最大下采样层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义批量归一化层
def batch_norm_layer(value, train=False, name='batch_norm'):
    if train is not False:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)
    else:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=False)


if __name__ == '__main__':
    net = MainNet()
    # 初始化参数值
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(10000000):
            x, y = mnist.train.next_batch(10)
            x = np.reshape(x, (-1, 28, 28, 1))
            _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.data: x, net.label: y})

            if epoch % 10 == 0:
                # print(_loss)
                test_data, test_label = mnist.test.next_batch(10)
                test_data = np.reshape(test_data, (-1, 28, 28, 1))
                test_output = sess.run(net.linear_layer2, feed_dict={net.data: test_data})

                test_y = np.argmax(test_data, axis=1)
                test_out = np.argmax(test_output, axis=1)
                print(np.mean(np.array(test_y == test_out, dtype=np.float32)))
