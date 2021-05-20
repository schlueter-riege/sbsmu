"""Exponentiated gradient descent for I-divergence"""

import numpy as np
from scipy.sparse import csr_matrix

from src.helpers import calculate_Z, Timer, TINY_CONST, random_W


def EGD(X, rank, time_limit=1.0, loss_calc_intvl=0.01, omega=0.1, seed=None):
    """
    Exponentiated gradient descent with the I-divergence for symmetric NMF

    :param X:               Matrix to factorize
    :param rank:            Rank of factor matrix
    :param time_limit:      Factorization time-limit in seconds
    :param loss_calc_intvl: Interval between checkpoints where loss is calculated or W is copied
    :param omega:           Over-relaxation parameter
    :param seed:            Random seed

    :return W_list:         List with copied versions of the factor matrix
    :return times:         List of times at which the factor matrix was copied
    :return iter_counts:         Number of completed iterations
    """
    np.random.seed(seed)
    size = X.shape[0]
    W = random_W(size, rank, seed)

    iter_counts = [0]
    n_iter = 0
    W_list = [W.copy()]
    times = [0]
    timer = Timer(time_limit, loss_calc_intvl)

    while not timer.time_limit_exceeded():
        # Perform exponentiated gradient update
        Z = calculate_Z(X, W)
        W_sum = W.sum(axis=0) + TINY_CONST
        grad = W_sum - Z.dot(W)
        lr = omega / W_sum
        W *= np.exp(-lr * grad)

        # Check time limit, update logs
        n_iter += 1
        if timer.is_checkpoint():
            timer.checkpoint_inc()
            times.append(timer.time)
            W_list.append(W.copy())
            iter_counts.append(n_iter)
    return W_list, times, iter_counts


# Compile update func
EGD(csr_matrix((5, 5)), 2, time_limit=1.0)
