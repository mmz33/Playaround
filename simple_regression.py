import numpy as np

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x);
    return 1 / (1 + np.exp(-x))

# input; (4, 3)
x = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = [[0], [1], [1], [0]] # output; (4, 1)

np.random.seed(1)

# create random weights
w1 = np.random.random((3, 4))
w2 = np.random.random((4, 1))

# training steps
for j in range(10000):

    # forward pass
    l0 = x
    l1 = sigmoid(np.dot(l0, w1)) # (4, 4)
    l2 = sigmoid(np.dot(l1, w2)) # (4, 1)

    l2_error = y - l2 # (4, 1)

    if j % 1000 == 0:
        print('Error: {}'.format(np.mean(np.abs(l2_error))))

    # backward pass
    l2_delta = l1.dot(l2_error * sigmoid(l2, deriv=True)) # (4, 1)
    l1_error = w2.dot(l2_delta.T) # (4, 4)
    l1_delta = l0.dot(l1_error * sigmoid(l1, deriv=True)) # (3, 4)

    w1 += l1_delta
    w2 += l2_delta

print('Prediction:\n', l2)
