import numpy as np


def mop2(x):

    n = x.shape[1]
    f1 = 1 - np.exp(-1 * np.sum(np.square(x - 1 / np.sqrt(n))))
    f2 = 1 - np.exp(-1 * np.sum(np.square(x + 1 / np.sqrt(n))))

    out = np.array([f1, f2])
    return out
