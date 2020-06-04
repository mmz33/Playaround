import tensorflow.compat.v1 as tf
from tensorflow.python.ops.nn import rnn_cell
import numpy as np

class MogrifierLSTM(rnn_cell.RNNCell):

    # Implementation of Migrofier LSTM
    # Paper: https://arxiv.org/abs/1909.01792
    # updates inspired from here: 
    #   https://github.com/deepmind/lamb/blob/254a0b0e330c44e00cf535f98e9538d6e735750b/lamb/tiled_lstm.py

    def __init__(self, num_units, rank=0, rounds=4, activation=tf.tanh):
        super(MogrifierLSTM, self).__init__()
        self.num_units = num_units
        self.rank = rank 
        self.rounds = rounds
        self.activation = activation

    @property
    def state_size(self):
        return rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    @staticmethod
    def linear(inputs, out_dim, with_bias=True, bias_init=0.0, name=None):
        assert name is not None
        input_dim = inputs.get_shape()[-1].value
        input_dtype = inputs.dtype
        weights = tf.get_variable(name=name + "_W", shape=[input_dim, out_dim],
                                  dtype=input_dtype)
        out = tf.matmul(inputs, weights) # (B, D')
        if with_bias:
            b = tf.get_variable(name=name + "_b", shape=[out_dim],
                dtype=input_dtype, initializer=tf.constant_initializer(bias_init))
            out += b
        return out
    
    @staticmethod
    def get_low_rank_matrices(shape, rank, name): 
        a = tf.get_variable(name + "_a", [shape[0], rank])
        b = tf.get_variable(name + "_b", [rank, shape[1]])
        return a, b
        
    def mogrify(self, x, h, rank):
        # x_i = 2 sig(Q_i h_{i - 1}) . x_{i - 2}    for odd i  [1...r]
        # h_i = 2 sig(R_i x_{i - 1}) . h_{i - 2}    for even i [1...r]

        # round 1: x_1 = F(h_0) * x_{-1}
        # round 2: h_2 = F(x_1) * h_0
        # round 3: x_3 = F(h_2) * x_1
        # round 4: h_4 = F(x_3) * h_2
        # ...
       
        x_dim = x.get_shape()[-1].value
        h_dim = self.num_units

        # Full-rank projection
        for round_ in range(1, self.rounds + 1):
            x_round = round_ % 2 == 1
            scope_name = 'mogrify_%i' % round_
            if rank == 0:
                if x_round:
                    x *= 2 * tf.sigmoid(self.linear(h, x_dim, name=scope_name))
                else:
                    h = 2 * tf.sigmoid(self.linear(x, h_dim, name=scope_name))
            else:
                if x_round:
                    shape = [h_dim, x_dim]
                else:
                    shape = [x_dim, h_dim]
                a, b = self.get_low_rank_matrices(shape, rank, scope_name)
                bias = tf.get_variable(scope_name + "_bias", [shape[-1]],
                                       initializer=tf.constant_initializer(0.0))
                if x_round:
                    x *= 2 * tf.sigmoid(tf.matmul(tf.matmul(h, a), b))
                else:
                    h *= 2 * tf.sigmoid(tf.matmul(tf.matmul(x, a), b))
        return x, h

    def __call__(self, inputs, state):
        prev_c, prev_h = state
        modulated_inputs, mouldated_prev_h = self.mogrify(inputs, prev_h, self.rank)
        inputs_linear = self.linear(modulated_inputs, 4 * self.num_units, name="modulated_inputs")
        prev_h_linear = self.linear(mouldated_prev_h, 4 * self.num_units, with_bias=False,
                                    name="mouldated_prev_h")
        lstm_in = tf.add(inputs_linear, prev_h_linear)
        i, g, f, o = tf.split(lstm_in, num_or_size_splits=4, axis=-1)
        new_c = tf.sigmoid(f) * prev_c + tf.sigmoid(i) * self.activation(g)
        new_h = tf.sigmoid(o) * self.activation(new_c)
        return new_h, rnn_cell.LSTMStateTuple(new_c, new_h)


def generate_batch_data(batch_size, max_seq_len, feature_dim):
    seq_len_batch = []
    batch_data = []
    for i in range(batch_size):
        seq_len = np.random.randint(1, max_seq_len + 1)
        seq_len_batch.append(seq_len)
        seq = np.random.rand(seq_len, feature_dim)  # generate random seq 
        seq = np.pad(seq, ((0, max_seq_len - seq_len), (0, 0)))
        batch_data.append(seq)
    return batch_data, seq_len_batch


if __name__ == '__main__':
    B = 300  # batch size
    D = 100  # feature dim
    T = 50  # max time steps
    H = 1024  # LSTM hidden units

    X = tf.placeholder(tf.float32, [B, T, D])
    seq_len = tf.placeholder(tf.int32, [B])

    mogrifier_cell = MogrifierLSTM(num_units=H)
    outputs, states = tf.nn.dynamic_rnn(
       mogrifier_cell, X, seq_len, dtype=tf.float32)

    X_batch, seq_len_batch = generate_batch_data(B, T, D)

    print('trainable variables:')
    for v in tf.trainable_variables():
        print(v.name, v.shape)
    print()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_val, states_val = sess.run([outputs, states],
            feed_dict={X: X_batch, seq_len: seq_len_batch})

        print(outputs_val)
        print()
        print(states_val)
