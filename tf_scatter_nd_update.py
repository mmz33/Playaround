#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

# (3, 3, 5)
ref = tf.Variable(
    [[[1, 2, 3, 4, 5], [10, 10, 10, 10, 10], [6, 7, 8, 9, 10]],
     [[11, 12, 13, 14, 15], [11, 11, 11, 11, 11], [16, 17, 18, 19, 20]],
     [[21, 22, 23, 24, 25], [12, 12, 12, 12, 12], [26, 27, 28, 29, 30]]]
)

def get_squeezed_slice(input_, begin, size):
    return tf.squeeze(tf.slice(input_, begin, size))

indices = [[0, 0], [0, 2]]  # num_updates=2, index_depth=2 (B,T)

# num_updates=2, index_depth=1 (D,)
updates = [
    get_squeezed_slice(ref, [0, 2, 0], [1, 1, 5]),
    get_squeezed_slice(ref, [0, 0, 0], [1, 1, 5])
]

res = tf.scatter_nd_update(ref, indices, updates)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('updates:', sess.run(updates))
    print(sess.run(ref))
    print(sess.run(res))

