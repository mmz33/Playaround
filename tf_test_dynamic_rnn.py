import tensorflow as tf
import numpy as np

D = 3 # feature dim
T = 3 # max time steps
H = 5 # LSTM hidden units

X = tf.placeholder(tf.float32, [None, T, D])
seq_len = tf.placeholder(tf.int32, [None])

basic_cell = tf.keras.layers.SimpleRNNCell(units=H)
layer = tf.keras.layers.RNN(basic_cell, return_state=True, return_sequences=True)
output, final_states = layer(X)

X_batch = np.array([
  [[0, 1, 2], [9, 8, 7], [0, 0, 0]],
  [[3, 4, 5], [0, 0, 0], [0, 0, 0]], # add zero padding
  [[6, 7, 8], [6, 5, 4], [0, 0, 0]],
  [[9, 0, 1], [3, 2, 1], [0, 0, 0]],
])

seq_len_batch = [2, 1, 2, 2]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output, final_states = sess.run([output, final_states],
        feed_dict={X: X_batch, seq_len: seq_len_batch})

    print(output)
    print()
    print(final_states)
