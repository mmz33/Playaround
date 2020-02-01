import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
import numpy as np

class MigrofierLSTM(rnn_cell.RNNCell):

    # Implementation of Migrofier LSTM
    # https://arxiv.org/abs/1909.01792

    def __init__(self, num_units, rounds=4, activation=tf.tanh):
        super(MigrofierLSTM, self).__init__()
        self.num_units = num_units
        self.rounds = rounds
        self.activation = activation

    @property
    def state_size(self):
        return rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    @staticmethod
    def linear(inputs, out_dim, with_bias=False, name=None):
        assert name is not None
        input_dim = inputs.get_shape().dims[-1].value
        weights = tf.get_variable(name="W_" + name, shape=[input_dim, out_dim],
            dtype=tf.float32)
        out = tf.matmul(inputs, weights) # (B, 4 * D)
        if with_bias:
            b = tf.get_variable(name="b_" + name, shape=[out_dim],
                dtype=tf.float32, initializer=tf.zeros_initializer())
            out += b
        return out

    def mogrify(self, curr_inp, prev_h):
        # x_i = 2 sig(Q_i h_{i - 1}) . x_{i - 2}    for odd i  [1...r]
        # h_i = 2 sig(R_i x_{i - 1}) . h_{i - 2}    for even i [1...r]

        # round 1: x_1 = F(h_0) * x_{-1}
        # round 2: h_2 = F(x_1) * h_0
        # round 3: x_3 = F(h_2) * x_1
        # round 4: h_4 = F(x_3) * h_2
        # ...
        # Thus, we only need to keep track of two variables and update them
        # after each computation

        inp_dim = curr_inp.get_shape().dims[-1].value
        h_dim = self.num_units

        curr_x = curr_inp # x_{-1}
        curr_h = prev_h # h_0
        for round in range(1, self.rounds + 1):
            if round % 2 == 1:
                x = 2 * tf.sigmoid(self.linear(curr_h, inp_dim, name="round_%i" % round)) * curr_x
                curr_x = x
            else:
                h = 2 * tf.sigmoid(self.linear(curr_x, h_dim, name="round_%i" % round)) * curr_h
                curr_h = h
        return curr_x, curr_h

    def __call__(self, inputs, state):
        prev_c, prev_h = state
        modulated_inputs, mouldated_prev_h = self.mogrify(inputs, prev_h)
        inputs_linear = self.linear(modulated_inputs, 4 * self.num_units, name="modulated_inputs")
        prev_h_linear = self.linear(mouldated_prev_h, 4 * self.num_units, name="mouldated_prev_h")
        lstm_in = tf.add(inputs_linear, prev_h_linear)
        i, g, f, o = tf.split(lstm_in, num_or_size_splits=4, axis=-1)
        new_c = tf.sigmoid(f) * prev_c + tf.sigmoid(i) * self.activation(g)
        new_h = tf.sigmoid(o) * self.activation(new_c)
        return new_h, rnn_cell.LSTMStateTuple(new_c, new_h)

if __name__ == '__main__':
    D = 3 # feature dim
    T = 3 # max time steps
    H = 3 # LSTM hidden units

    X = tf.placeholder(tf.float32, [None, T, D])
    seq_len = tf.placeholder(tf.int32, [None])

    mogrifier_cell = MigrofierLSTM(num_units=H)
    outputs, states = tf.nn.dynamic_rnn(
       mogrifier_cell, X, seq_len, dtype=tf.float32)

    # (B, T, D)
    X_batch = np.array([
      [[0, 1, 2], [9, 8, 7], [0, 0, 0]],
      [[3, 4, 5], [0, 0, 0], [0, 0, 0]], # add zero padding
      [[6, 7, 8], [6, 5, 4], [0, 0, 0]],
      [[9, 0, 1], [3, 2, 1], [0, 0, 0]],
    ])

    seq_len_batch = [2, 1, 2, 2]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_val, states_val = sess.run([outputs, states],
            feed_dict={X: X_batch, seq_len: seq_len_batch})

        print(outputs_val)
        print()
        print(states_val)
