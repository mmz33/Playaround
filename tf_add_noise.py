#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

B = 10 
T = 1
D = 5

def add_gauss_noise(inputs):
    guassian_noise = tf.Variable(tf.random.normal(shape=tf.shape(inputs)), trainable=False)
    return inputs + guassian_noise

class RandomBernoulli:
    def __init__(self, p):
        self.p = p

    def sample_n(self, n):
        uniform_dist = tf.random.uniform(shape=[n])
        samples = tf.math.less(uniform_dist, self.p)
        return tf.cast(samples, tf.bool) 

def add_seq_noise(inputs):
    random_bernoulli = RandomBernoulli(p=0.2)
    samples = random_bernoulli.sample_n(tf.shape(inputs)[0])
    noise = tf.Variable(tf.random.normal(shape=tf.shape(inputs)), trainable=False)
    res = tf.where(samples, noise + inputs, inputs)
    return res, samples

x = tf.Variable(tf.random.uniform(shape=(B, T, D)), trainable=False)

with tf.Session() as sess:
    noisy_x, samples = add_seq_noise(x)
    sess.run(tf.global_variables_initializer())
    print('original x\n', sess.run(x))
    print('samples\n', sess.run(samples))
    print('Adding guassian noise:')
    print(sess.run(noisy_x))
