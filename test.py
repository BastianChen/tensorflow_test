import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
#
# class MLPNet:
#
#     def __init__(self):
#         self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
#         self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
#
#         self.in_w = tf.Variable(tf.truncated_normal(shape=[784, 100], stddev=0.1))
#         self.in_b = tf.Variable(tf.zeros([100]))
#
#         self.out_w = tf.Variable(tf.truncated_normal(shape=[100, 10], stddev=0.1))
#         self.out_b = tf.Variable(tf.zeros([10]))
#
#         self.forward()
#         self.backward()
#
#     def forward(self):
#         self.fc1 = tf.nn.relu(tf.matmul(self.x, self.in_w) + self.in_b)
#         self.output = tf.nn.softmax(tf.matmul(self.fc1, self.out_w) + self.out_b)
#         pass
#
#     def backward(self):
#         self.loss = tf.reduce_mean((self.output - self.y) ** 2)
#         self.opt = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
#
#     def test(self):
#         pass
#
#
# if __name__ == '__main__':
#     net = MLPNet()
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#
#         for epoch in range(10000000):
#             xs, ys = mnist.train.next_batch(100)
#
#             _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x: xs, net.y: ys})
#
#             if epoch % 100 == 0:
#                 # print(_loss)
#                 test_xs, test_ys = mnist.test.next_batch(10000)
#                 test_output = sess.run(net.output, feed_dict={net.x: test_xs})
#
#                 test_y = np.argmax(test_ys, axis=1)
#                 test_out = np.argmax(test_output, axis=1)
#                 print(np.mean(np.array(test_y == test_out, dtype=np.float32)))

# 定义张量
a = tf.constant([1, 2])
b = tf.constant([3, 4])
# 建议了联系关系，但没有实现数据流通
c = a + b
print(c)

d = tf.Variable([0])
# 给参数赋值
e = tf.assign(d, [2])
# 使用tf.global_variables_initializer()添加节点用于初始化所有的变量
init = tf.global_variables_initializer()

# 定义变量
f = tf.placeholder(dtype=tf.float32, shape=[1])
with tf.Session() as sess:
    # 初始化所有参数
    sess.run(init)
    c, d = sess.run([c, f], feed_dict={f: [2]})
    print(c)
    print(d)
    a, b = sess.run([a, b])
    print(a)
    print(b)
