"""Stochastic Bound-and-Scale Multiplicative Updates for I-divergence"""

import multiprocessing as mp
import os
from ctypes import c_double, c_bool, c_int64

import numba as nb
import numpy as np
from scipy.sparse import csr_matrix

from src.helpers import TINY_CONST, INT32_MAX, random_W, Timer, nb_parse_csr

CPU_COUNT = os.cpu_count()
SHARED_FACTOR_ARR = None
SHARED_ITER_ARR = None
SHARED_READY_SIGNAL = None
SHARED_STOP_SIGNAL = None


def PSGD(X, rank, eta=0.1, time_limit=1.0, loss_calc_intvl=0.01, pool=None, seed=None):
    """
    Projected stochastich gradient descent with the I-divergence for symmetric NMF

    :param X:               Matrix to factorize
    :param rank:            Rank of factor matrix
    :param eta:             Learning rate
    :param time_limit:      Factorization time-limit in seconds, excluding worker initialization
    :param loss_calc_intvl: Interval between checkpoints where loss is calculated or W is copied
    :param pool:            Multiprocessing pool
    :param seed:            Random seed

    :return W_list:         List with copied versions of the factor matrix
    :return times:          List of times at which the factor matrix was copied
    :return iter_counts:    Number of completed iterations
    """
    size = X.shape[0]

    # Initialize a pool of worker processes if not provided
    if pool is None:
        close_pool = True
        pool = init_pool(size, rank)
    else:
        close_pool = False

    np.random.seed(seed)

    # Initialize shared variables
    W = _shared_to_ndarray(SHARED_FACTOR_ARR, c_double, shape=(size, rank))
    W[:] = random_W(size, rank, seed)
    stop_signal = _shared_to_ndarray(SHARED_STOP_SIGNAL, c_bool)
    stop_signal[0] = False
    with SHARED_READY_SIGNAL.get_lock():
        ready_signal = _shared_to_ndarray(SHARED_READY_SIGNAL.get_obj(), c_bool)
        ready_signal[0] = False

    # Dispatch workers
    worker_seed = np.random.randint(INT32_MAX)
    results = pool.apply_async(_factorize, args=(X, rank, eta, worker_seed))

    # Wait for timer and gather results
    W_list, times, iter_counts = _timer(W, time_limit, loss_calc_intvl)
    stop_signal[0] = True
    results.get()

    # Close worker pool if initialized within process
    if close_pool:
        pool.close()

    return W_list, times, iter_counts


@nb.njit(fastmath=True)
def _nb_factorize(X, W, stop_signal, eta, iter_arr, seed):
    """
    Numba-compiled symmetric factorization of X using the I-divergence. Runs until signals[0] is false

    :param X:            Matrix to factorize
    :param W:            Shared factor matrix
    :param stop_signal:  Signals for process to quit
    :param eta:          Learning rate
    :param seed:         Random seed

    :return:             Number of iterations
    """
    np.random.seed(seed)
    size, rank = W.shape
    n_iter = 0
    projection_array = np.full_like(W[0], TINY_CONST)
    
    while True:
        # Generate lists of random indices, which is faster than calling np.random every iteration
        i_indices = np.random.randint(0, size, 10000)
        j_indices = np.random.randint(0, size, 10000)

        for i, j in zip(i_indices, j_indices):
            # Store copies of rows of W in memory to avoid concurrency issues
            W_i = W[i].copy()
            W_j = W[j].copy()

            # Stop the factorization if it diverges
            for k in range(rank):
                if W_i[k] > INT32_MAX or W_j[k] > INT32_MAX:
                    return n_iter

            # Handle X_ij == 0
            if not (i, j) in X:
                gradient_i = 2 * W_j
                gradient_j = 2 * W_i
            # Handle X_ij != 0
            else:
                z = X[(i, j)] / (W_i @ W_j.T + TINY_CONST)
                gradient_i = 2 * (1 - z) * W_j
                gradient_j = 2 * (1 - z) * W_i

            # Perform update
            W[i] = np.maximum(W_i - eta * gradient_i, projection_array)
            W[j] = np.maximum(W_j - eta * gradient_j, projection_array)

            iter_arr[0] += 1

            if stop_signal[0]:
                return n_iter


def _factorize(X, rank, eta, seed):
    """Parse shared variables and X, start Numba-compiled factorization"""
    size = X.shape[0]
    stop_signal = _shared_to_ndarray(SHARED_STOP_SIGNAL, c_bool)
    W = _shared_to_ndarray(SHARED_FACTOR_ARR, c_double, shape=(size, rank))
    iter_arr = _shared_to_ndarray(SHARED_ITER_ARR, c_int64)
    X = nb_parse_csr(X)

    # Signal worker ready
    with SHARED_READY_SIGNAL.get_lock():
        signal = _shared_to_ndarray(SHARED_READY_SIGNAL.get_obj(), c_bool)
        signal[0] = True

    return _nb_factorize(X, W, stop_signal, eta, iter_arr, seed)


def _init_worker(shared_factor_arr, shared_iter_arr, shared_ready_signal, shared_stop_signal):
    """Load global variables into worker process"""
    global SHARED_FACTOR_ARR
    global SHARED_ITER_ARR
    global SHARED_READY_SIGNAL
    global SHARED_STOP_SIGNAL
    SHARED_FACTOR_ARR = shared_factor_arr
    SHARED_ITER_ARR = shared_iter_arr
    SHARED_READY_SIGNAL = shared_ready_signal
    SHARED_STOP_SIGNAL = shared_stop_signal


def init_pool(size, rank):
    """Initialize multiprocessing pool and shared variables"""
    global SHARED_FACTOR_ARR
    global SHARED_ITER_ARR
    global SHARED_READY_SIGNAL
    global SHARED_STOP_SIGNAL
    SHARED_FACTOR_ARR = mp.Array(c_double, size * rank, lock=False)
    SHARED_ITER_ARR = mp.Array(c_int64, 1, lock=False)
    SHARED_READY_SIGNAL = mp.Array(c_bool, 1, lock=True)
    SHARED_STOP_SIGNAL = mp.Array(c_bool, 1, lock=False)

    pool = mp.Pool(initializer=_init_worker, processes=1,
                   initargs=(SHARED_FACTOR_ARR, SHARED_ITER_ARR, SHARED_READY_SIGNAL, SHARED_STOP_SIGNAL))

    # Run PSGD on empty X to initialize worker processes
    PSGD(csr_matrix((size, size)), rank, pool=pool)
    return pool


def _timer(W, time_limit, loss_calc_interval):
    """Timer that saves copies of W at regular time intervals"""
    times = [0]
    iter_counts = [0]
    losses = [W.copy()]
    iter_arr = _shared_to_ndarray(SHARED_ITER_ARR, c_int64)

    # Wait for worker initialization to complete
    while True:
        with SHARED_READY_SIGNAL.get_lock():
            signal = _shared_to_ndarray(SHARED_READY_SIGNAL.get_obj(), c_bool)
            if signal[0]:
                break

    timer = Timer(time_limit, loss_calc_interval)
    while not timer.time_limit_exceeded():
        if timer.is_checkpoint():
            timer.checkpoint_inc()
            times.append(timer.time)
            iter_counts.append(iter_arr[0])
            losses.append(W.copy())

    return losses, times, iter_counts


def _shared_to_ndarray(shared_arr, dtype, shape=None):
    """Parses a shared multiprocessing Array object as a Numpy array"""
    arr = np.frombuffer(shared_arr, dtype=dtype)
    if shape is None:
        return arr
    return arr.reshape(shape)
