#!/usr/bin/env python3

import numpy as np

a = 1
b = 1
print(np.dot(a, b))

a = [1, 2, 3]
b = [2, 3, 4]
print(np.dot(a, b))

a = [[1, 2, 3], [1, 2, 3]] # (2, 3)
b = [[2, 3], [2, 3], [2, 3]] # (3, 2)
print(np.dot(a, b)) # (2, 2)

a = np.array([[[1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5]]]) # (2, 2, 3)
b = np.array([[[1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5]]]) # (2, 2, 3)
b = b.T # (3, 2, 2)
b = np.swapaxes(b, 0, -2)
print(b.shape)
res = np.dot(a, b)
print(res.shape)

