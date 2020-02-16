#!/usr/bin/env python3

import tensorflow as tf

# Testing CPU configuration in TensorFLow using "tf.ConfigProto"

with tf.device('/cpu:1'):
    x = tf.constant(10, name='x')
    y = tf.constant(2, name='y')
    z = tf.multiply(x, y, name='z')

# set CPU count to 2 since we are using 2 cpu devices otherwise an error will
# happen
config = tf.ConfigProto(log_device_placement=True, device_count={'CPU': 2})
with tf.Session(config=config) as sess:
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.add(a, b, 'c')
    print('c:', sess.run(c))
    print('z:', sess.run(z))


# Device Mapping:
# ===============
# z: (Mul): /job:localhost/replica:0/task:0/device:CPU:1
# 2020-02-16 23:36:01.069628: I tensorflow/core/common_runtime/placer.cc:54] z: (Mul)/job:localhost/replica:0/task:0/device:CPU:1
# c: (Add): /job:localhost/replica:0/task:0/device:CPU:0
# 2020-02-16 23:36:01.069641: I tensorflow/core/common_runtime/placer.cc:54] c: (Add)/job:localhost/replica:0/task:0/device:CPU:0
# x: (Const): /job:localhost/replica:0/task:0/device:CPU:1
# 2020-02-16 23:36:01.069649: I tensorflow/core/common_runtime/placer.cc:54] x: (Const)/job:localhost/replica:0/task:0/device:CPU:1
# y: (Const): /job:localhost/replica:0/task:0/device:CPU:1
# 2020-02-16 23:36:01.069655: I tensorflow/core/common_runtime/placer.cc:54] y: (Const)/job:localhost/replica:0/task:0/device:CPU:1
# a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# 2020-02-16 23:36:01.069662: I tensorflow/core/common_runtime/placer.cc:54] a: (Const)/job:localhost/replica:0/task:0/device:CPU:0
# b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# 2020-02-16 23:36:01.069668: I tensorflow/core/common_runtime/placer.cc:54] b: (Const)/job:localhost/replica:0/task:0/device:CPU:0

