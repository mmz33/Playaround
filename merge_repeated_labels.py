import tensorflow as tf

def remove_blank_and_merge_repeated_values(indices, blank_idx=-1):
    """
    :param tf.Tensor indices: ctc output indices of shape [T]
    :param int blank_index: ctc blank index
    """

    shape = tf.shape(indices)
    assert len(shape) == 1
    T = shape[0]
    shift_right = tf.slice(indices, [0], [T-1])
    shift_right = tf.concat([[-1], shift_right], axis=0)
    
    non_blank_mask = indices != blank_idx
    unique_mask = indices != shift_right
    mask = non_blank_mask & unique_mask

    res = tf.boolean_mask(indices, mask)

    return res


x = tf.constant([-1, 0, 0, -1, 2, 3, -1])
out = remove_blank_and_merge_repeated_values(x)
tf.debugging.assert_equal(out, tf.constant([0, 2, 3]))

x = tf.constant([0])
out = remove_blank_and_merge_repeated_values(x)
tf.debugging.assert_equal(out, tf.constant([0]))

x = tf.constant([1, 1, 1, 1])
out = remove_blank_and_merge_repeated_values(x)
tf.debugging.assert_equal(out, tf.constant([1]))

x = tf.constant([0, 1, 2, 3])
out = remove_blank_and_merge_repeated_values(x)
tf.debugging.assert_equal(out, tf.constant([0, 1, 2, 3]))

x = tf.constant([-1, 0, -1, 0, -1, 1, 2, -1])
out = remove_blank_and_merge_repeated_values(x)
tf.debugging.assert_equal(out, tf.constant([0, 0, 1, 2]))

x = tf.constant([0, -1, 0, 1])
out = remove_blank_and_merge_repeated_values(x)
tf.debugging.assert_equal(out, tf.constant([0, 0, 1]))


















