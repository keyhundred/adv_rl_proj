import numpy as np
from copy import deepcopy

filt_size = [3, 3]

r = np.random.randn(10, 10)
o = np.ones((10, 10))
f = np.array([
    [0, 2, 3,],
    [4, 5, 6,],
    [7, 8, 9,]
])

def make_filter(ones, filt, x, y):
    for i in range(filt_size[0]):
        for j in range(filt_size[1]):
            ones[x + i][y + j] = filt[i][j]
    return ones

print(filtering(o, f, 1, 1))