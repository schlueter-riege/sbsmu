"""Multiplicative Updates for I-divergence"""

import numpy as np
from scipy.sparse import csr_matrix

from src.helpers import Timer, TINY_CONST, calculate_Z, random_W


def MU(X, rank, time_limit=1.0, loss_calc_intvl=0.01, seed=None):
    """
    Full-Batch Multiplicative Updates with the I-divergence for symmetric NMF

    :param X:               Matrix to factorize
    :param rank:            Rank of factor matrix
    :param time_limit:      Factorization time-limit in seconds
    :param loss_calc_intvl: Interval between checkpoints where loss is calculated or W is copied
    :param seed:            Random seed

    :return W_list:         List with copied versions of the factor matrix
    :return times:          List of times at which the factor matrix was copied
    :return iter_counts:    Number of completed iterations
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
        # Perform multiplicative update
        Z = calculate_Z(X, W)
        grad_neg = Z.dot(W)
        grad_pos = W.sum(axis=0) + TINY_CONST
        W = W * np.sqrt(grad_neg / grad_pos)

        # Check time limit, update logs
        n_iter += 1
        if timer.is_checkpoint():
            timer.checkpoint_inc()
            times.append(timer.time)
            W_list.append(W.copy())
            iter_counts.append(n_iter)

    return W_list, times, iter_counts


# Compile update func
MU(csr_matrix((5, 5)), 2, time_limit=1.0)
