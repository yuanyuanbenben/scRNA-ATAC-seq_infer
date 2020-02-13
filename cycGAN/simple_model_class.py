# import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# simple situation
# just one kind of cell

# x -- RNAseq, y -- ATACseq
# x1,y2 true data, x2,y1 false data
# x1 -> y1 G:generator, Dy:discriminator
# y2 -> x1 F:generator, Dx:discriminator

class scRAP(object):
    """
    a simple situation version
    """

    def __init__(self, m, n):
        self._m = m
        self._n = n

    @staticmethod
    def lrelu(x, alpha=0.2):
        with tf.variable_scope('leakyRelu'):
            return tf.maximum(x, alpha * x)

    # generator G
    def G(self, Z, dim_Z):
        with tf.variable_scope("G", reuse=tf.AUTO_REUSE):
            dim_1 = 2 * dim_Z
            dim_2 = 2 * self._n
            dim_3 = self._n

            W1 = tf.get_variable("G_W1", shape=[dim_Z, dim_1],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W2 = tf.get_variable("G_W2", shape=[dim_1, dim_2],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W3 = tf.get_variable("G_W3", shape=[dim_2, dim_3],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            B1 = tf.get_variable("G_B1", shape=[dim_1], dtype=tf.float32, initializer=tf.constant_initializer())
            B2 = tf.get_variable("G_B2", shape=[dim_2], dtype=tf.float32, initializer=tf.constant_initializer())
            B3 = tf.get_variable("G_B3", shape=[dim_3], dtype=tf.float32, initializer=tf.constant_initializer())

            fc1 = self.lrelu(tf.add(tf.matmul(Z, W1), B1))
            fc2 = self.lrelu(tf.add(tf.matmul(fc1, W2), B2))
            fc3 = tf.nn.sigmoid(tf.add(tf.matmul(fc2, W3), B3))
            return fc3

    # generator F
    def F(self, Z, dim_Z):
        with tf.variable_scope("F", reuse=tf.AUTO_REUSE):
            dim_1 = 2 * dim_Z
            dim_2 = 2 * self._m
            dim_3 = self._m

            W1 = tf.get_variable("F_W1", shape=[dim_Z, dim_1],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W2 = tf.get_variable("F_W2", shape=[dim_1, dim_2],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W3 = tf.get_variable("F_W3", shape=[dim_2, dim_3],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            B1 = tf.get_variable("F_B1", shape=[dim_1], dtype=tf.float32, initializer=tf.constant_initializer())
            B2 = tf.get_variable("F_B2", shape=[dim_2], dtype=tf.float32, initializer=tf.constant_initializer())
            B3 = tf.get_variable("F_B3", shape=[dim_3], dtype=tf.float32, initializer=tf.constant_initializer())

            fc1 = self.lrelu(tf.add(tf.matmul(Z, W1), B1))
            fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, W2), B2))
            fc3 = tf.add(tf.matmul(fc2, W3), B3)
            return fc3

    # discriminator Dy
    def Dy(self, Y):
        with tf.variable_scope("Dy", reuse=tf.AUTO_REUSE):
            dim_1 = self._n // 2
            dim_2 = self._n // 4
            dim_3 = 1

            W1 = tf.get_variable("Dy_W1", shape=[self._n, dim_1],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W2 = tf.get_variable("Dy_W2", shape=[dim_1, dim_2],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W3 = tf.get_variable("Dy_W3", shape=[dim_2, dim_3],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            B1 = tf.get_variable("Dy_B1", shape=[dim_1], dtype=tf.float32, initializer=tf.constant_initializer())
            B2 = tf.get_variable("Dy_B2", shape=[dim_2], dtype=tf.float32, initializer=tf.constant_initializer())
            B3 = tf.get_variable("Dy_B3", shape=[dim_3], dtype=tf.float32, initializer=tf.constant_initializer())

            fc1 = self.lrelu(tf.add(tf.matmul(Y, W1), B1))
            fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, W2), B2))
            fc3 = tf.add(tf.matmul(fc2, W3), B3)
            return fc3, tf.nn.sigmoid(fc3)

    # discriminator Dx
    def Dx(self, X):
        with tf.variable_scope("Dy", reuse=tf.AUTO_REUSE):
            dim_1 = self._m // 2
            dim_2 = self._m // 4
            dim_3 = 1

            W1 = tf.get_variable("Dx_W1", shape=[self._m, dim_1],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W2 = tf.get_variable("Dx_W2", shape=[dim_1, dim_2],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            W3 = tf.get_variable("Dx_W3", shape=[dim_2, dim_3],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            B1 = tf.get_variable("Dx_B1", shape=[dim_1], dtype=tf.float32, initializer=tf.constant_initializer())
            B2 = tf.get_variable("Dx_B2", shape=[dim_2], dtype=tf.float32, initializer=tf.constant_initializer())
            B3 = tf.get_variable("Dx_B3", shape=[dim_3], dtype=tf.float32, initializer=tf.constant_initializer())

            fc1 = self.lrelu(tf.add(tf.matmul(X, W1), B1))
            fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, W2), B2))
            fc3 = tf.add(tf.matmul(fc2, W3), B3)
            return fc3, tf.nn.sigmoid(fc3)

