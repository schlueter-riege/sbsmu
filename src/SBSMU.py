"""Stochastic Bound-and-Scale Multiplicative Updates for I-divergence"""
import multiprocessing as mp
import os
from ctypes import c_double, c_bool, c_int64

import numba as nb
import numpy as np
from scipy.sparse import csr_matrix

from src.helpers import TINY_CONST, INT32_MAX, i_divergence, random_W, Timer

CPU_COUNT = os.cpu_count()
SHARED_FACTOR_ARR = None
SHARED_ITER_ARR = None
SHARED_READY_SIGNAL = None
SHARED_STOP_SIGNAL = None


def SBSMU(X, rank, eta=0.8, alpha=0.999999, beta=0.25, time_limit=1.0, loss_calc_intvl=0.01, return_losses=True,
          W_init=None, pool=None, seed=None):
    """
    Stochastic Bound-and-Scale Multiplicative Updates with the I-divergence for symmetric NMF. If return_losses is
    False, the algorithm will not detect convergence and will simply run until the time limit.

    :param X:               Matrix to factorize
    :param rank:            Rank of factor matrix
    :param eta:             Exponent learning rate
    :param alpha:           Bound-and-Scale variable
    :param beta:            Separate sampling threshold
    :param time_limit:      Factorization time-limit in seconds, excluding worker initialization
    :param loss_calc_intvl: Interval between checkpoints where loss is calculated or W is copied
    :param return_losses:   If True calculate and return losses, else return copies of W
    :param W_init:          Pre-initialized factor matrix
    :param pool:            Multiprocessing pool
    :param seed:            Random seed

    :return W_list:         List with copied versions of the factor matrix
    :return t_list:         List of times at which the factor matrix was copied
    :return n_iter:         Number of completed iterations
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
    if W_init is None:
        W[:] = random_W(size, rank, seed)
    else:
        W[:] = W_init

    stop_signal = _shared_to_ndarray(SHARED_STOP_SIGNAL, c_bool)
    stop_signal[0] = False
    with SHARED_READY_SIGNAL.get_lock():
        ready_signal = _shared_to_ndarray(SHARED_READY_SIGNAL.get_obj(), c_bool)
        ready_signal[0] = False

    # Dispatch workers
    worker_seeds = np.random.randint(INT32_MAX, size=CPU_COUNT)
    results = [pool.apply_async(_factorize, args=(X, rank, eta, alpha, beta, i, s)) for i, s in enumerate(worker_seeds)]

    # Wait for timer and gather results
    losses, times, iter_counts = _timer(X, W, time_limit, loss_calc_intvl, return_losses)
    stop_signal[0] = True
    for res in results:
        res.get()

    # Close worker pool if initialized within process
    if close_pool:
        pool.close()

    return losses, times, iter_counts


@nb.njit(fastmath=True)
def _nb_nonzero_cycler(i_arr, j_arr, x_arr, seed):
    """Numba-compiled cyclic generator of random indices of nonzero-entries in X, distributed according to X"""
    np.random.seed(seed)
    cumulative_distribution = np.cumsum(x_arr)
    while True:
        samples = np.random.rand(100000)
        indices = np.searchsorted(cumulative_distribution, samples, side='right')
        for idx in indices:
            yield i_arr[idx], j_arr[idx]


@nb.njit(fastmath=True)
def _nb_zero_cycler(size, seed):
    """Numba-compiled cyclic generator of random indices of zero-entries in X"""
    np.random.seed(seed)
    while True:
        i_indices = np.random.randint(0, size, 100000)
        j_indices = np.random.randint(0, size, 100000)
        for idx in range(100000):
            yield i_indices[idx], j_indices[idx]


@nb.njit(fastmath=True)
def _nb_factorize(i_arr, j_arr, x_arr, W, stop_signal, eta, alpha, beta, iter_arr, worker_id, seed):
    """
    Numba-compiled symmetric factorization of X using the I-divergence. Runs until signals[0] is false

    :param i_arr:        Nonzero i-indices of X
    :param j_arr:        Nonzero j-indices of X
    :param x_arr:        Nonzero values of X
    :param W:            Shared factor matrix
    :param stop_signal:  Signals for process to quit
    :param eta:          Exponent learning rate
    :param alpha:        Bound-and-Scale variable
    :param beta:         Separate sampling threshold
    :param iter_arr:     Shared array for keeping track iteration count
    :param worker_id:    Worker ID
    :param seed:         Random seed

    :return:             Number of iterations
    """
    np.random.seed(seed)
    size, rank = W.shape
    loop_size = 1000
    b_vals = np.random.rand(loop_size)
    zero_grad_factor = (beta / (1 - beta)) * size ** 2
    nonzero_cycler = _nb_nonzero_cycler(i_arr, j_arr, x_arr, seed)
    zero_cycler = _nb_zero_cycler(size, seed)

    # Allow compilation with empty X
    if x_arr.shape[0] == 0:
        return

    while True:
        for b in b_vals:
            if b < beta:
                i, j = next(nonzero_cycler)

                # Store copies of rows of W in memory to avoid concurrency issues
                W_i = W[i].copy()
                W_j = W[j].copy()

                z = (1 - alpha) * 2 / (W_i @ W_j.T + TINY_CONST)
                W[i] = W_i * (alpha + z * W_j / alpha) ** eta
                W[j] = W_j * (alpha + z * W_i / alpha) ** eta

            else:
                i, j = next(zero_cycler)

                # Store copies of rows of W in memory to avoid concurrency issues
                W_i = W[i].copy()
                W_j = W[j].copy()

                W[i] = W_i * (alpha / (alpha + (1 - alpha) * 2 * zero_grad_factor * W_j)) ** eta
                W[j] = W_j * (alpha / (alpha + (1 - alpha) * 2 * zero_grad_factor * W_i)) ** eta

        iter_arr[worker_id] += loop_size
        if stop_signal[0]:
            return


def _factorize(X, rank, eta, alpha, beta, worker_id, seed):
    """Parse shared variables and X, start Numba-compiled factorization"""
    size = X.shape[0]
    stop_signal = _shared_to_ndarray(SHARED_STOP_SIGNAL, c_bool)
    W = _shared_to_ndarray(SHARED_FACTOR_ARR, c_double, shape=(size, rank))
    iter_arr = _shared_to_ndarray(SHARED_ITER_ARR, c_int64)
    i_arr, j_arr = X.nonzero()

    # Signal worker ready
    with SHARED_READY_SIGNAL.get_lock():
        signal = _shared_to_ndarray(SHARED_READY_SIGNAL.get_obj(), c_bool)
        signal[0] = True

    return _nb_factorize(i_arr, j_arr, X.data, W, stop_signal, eta, alpha, beta, iter_arr, worker_id, seed)


def _shared_to_ndarray(shared_arr, dtype, shape=None):
    """Parses a shared multiprocessing Array object as a Numpy array"""
    arr = np.frombuffer(shared_arr, dtype=dtype)
    if shape is None:
        return arr
    return arr.reshape(shape)


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
    SHARED_ITER_ARR = mp.Array(c_int64, CPU_COUNT, lock=False)
    SHARED_READY_SIGNAL = mp.Array(c_bool, 1, lock=True)
    SHARED_STOP_SIGNAL = mp.Array(c_bool, 1, lock=False)

    pool = mp.Pool(initializer=_init_worker,
                   initargs=(SHARED_FACTOR_ARR, SHARED_ITER_ARR, SHARED_READY_SIGNAL, SHARED_STOP_SIGNAL))

    # Run SBSMU on empty X to initialize worker processes
    SBSMU(csr_matrix((size, size)), rank, pool=pool)
    return pool


def _timer(X, W, time_limit, loss_calc_interval, return_losses):
    """Timer that saves copies of W at regular time intervals"""
    times = [0]
    iter_counts = [0]
    losses = [W.copy() if not return_losses else i_divergence(X, W.copy())]
    min_loss = np.inf
    n_checkpoints_loss_inc = 0
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
            iter_counts.append(sum(iter_arr))

            W_copy = W.copy()
            if return_losses:
                loss = i_divergence(X, W_copy)
                losses.append(loss)

                if loss < min_loss:
                    min_loss = loss
                    n_checkpoints_loss_inc = 0
                elif n_checkpoints_loss_inc >= 50:
                    break
                else:
                    n_checkpoints_loss_inc += 1
            else:
                losses.append(W_copy)

    return losses, times, iter_counts
