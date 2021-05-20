"""Functions and methods used by multiple classes"""
from pathlib import Path

import numba as nb
import numpy as np
from scipy.io import loadmat
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
from timeit import default_timer

TINY_CONST = 1e-16
INT32_MAX = 2147483647
SYMMETRIC_DATASETS = [
    'dolphins',
    'football',
    'korea1',
    'korea2',
    'strike'
]


def random_W(size, rank, seed):
    """Randomly initialized W with entries in (0,1]"""
    np.random.seed(seed)
    W = 1 - np.random.rand(size, rank)
    return np.abs(W)


def load_matrix(path, labels=False):
    """Load the dataset, ensure that it is symmetric, normalize it, find the rank and parse it as a dok_matrix"""
    path = Path(path)
    dataset = loadmat(path)
    X = dataset['A']
    label_arr = dataset['C']
    rank = np.unique(label_arr).shape[0]
    X = csr_matrix(X)
    X.sort_indices()

    # Ensure symmetry
    if path.stem not in SYMMETRIC_DATASETS:
        X = 0.5 * (X + X.T)

    # Normalize
    X /= X.sum()

    if labels:
        return X, rank, label_arr
    return X, rank


class Timer:
    """Keep track of time limit and checkpoints"""

    def __init__(self, time_limit, loss_calc_intvl):
        self.time_limit = time_limit
        self.loss_calc_intvl = loss_calc_intvl
        self.next_checkpoint = 0
        self.start_time = default_timer()

    @property
    def time(self):
        return default_timer() - self.start_time

    def is_checkpoint(self):
        return self.time > self.next_checkpoint

    def checkpoint_inc(self):
        self.next_checkpoint += self.loss_calc_intvl

    def time_limit_exceeded(self):
        return self.time > self.time_limit


@nb.njit(fastmath=True)
def _calculate_Z(i_arr, j_arr, x_arr, W):
    """Numba-compiled computation of Z"""
    n = x_arr.shape[0]
    z_arr = np.empty_like(x_arr)
    for index in range(n):
        x = x_arr[index]
        i = i_arr[index]
        j = j_arr[index]
        z = x / (W[i] @ W[j].T + TINY_CONST)
        z_arr[index] = z
    return z_arr


def calculate_Z(X, W):
    """
    Compute Z=X/(W*W^T) efficiently for sparse X

    :param X: Sparse input data as csr-matrix
    :param W: Factor matrix as ndarray
    """
    i_arr, j_arr = X.nonzero()
    z_arr = _calculate_Z(i_arr, j_arr, X.data, W)
    return coo_matrix((z_arr, (i_arr, j_arr)), shape=X.shape).tocsr()


@nb.njit(fastmath=True)
def _i_divergence(i_arr, j_arr, x_arr, W):
    """Calculate the I-divergence, Numba-compiled"""
    kl_div = 0
    for k in range(W.shape[1]):
        kl_div += np.sum(W[:, k]) ** 2

    for index in range(x_arr.shape[0]):
        i = i_arr[index]
        j = j_arr[index]
        if j > i:
            continue

        x = x_arr[index]
        x_r = W[i] @ W[j].T
        if j == i:
            kl_div += x * np.log((x / (x_r + TINY_CONST)) + TINY_CONST) - x
        else:
            kl_div += 2 * (x * np.log((x / (x_r + TINY_CONST)) + TINY_CONST) - x)

    return kl_div


def i_divergence(X, W):
    """Calculate the I-divergence"""
    i_arr, j_arr = X.nonzero()
    return _i_divergence(i_arr, j_arr, X.data, W)


def nb_parse_csr(X):
    """Convert csr-matrix X to a numba typed Dict"""
    nb_X = nb.typed.Dict.empty(
        nb.types.UniTuple(nb.types.int64, count=2),
        nb.types.float64
    )
    i_arr, j_arr = X.nonzero()
    for i, j in zip(i_arr, j_arr):
        nb_X[(i, j)] = X[i, j]
    return nb_X
