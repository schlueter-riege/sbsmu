import multiprocessing as mp
import pickle
from timeit import default_timer
import datetime
from pathlib import Path
import sys

from src.PSGD import PSGD
from src.MU import MU
from src.SBSMU import SBSMU
from src.EGD import EGD
from src.helpers import i_divergence, load_matrix


ALGOS = {
    'PSGD': {'algorithm': PSGD, 'config': {'eta': 1.0e-5}},
    'MU': {'algorithm': MU, 'config': {}},
    'EGD': {'algorithm': EGD, 'config': {'omega': 0.1}},
    'SBSMU': {'algorithm': SBSMU, 'config': {'eta': 0.8, 'alpha': 0.999999, 'beta': 0.25, 'return_losses': False}}
}

EXPERIMENTS = {
    'mnist': {
        'algos': ['PSGD', 'MU', 'EGD', 'SBSMU'],
        'time_limit': 601,
        'loss_calc_intvl': 1,
        'dataset': 'mnist_70k_10nn.mat',
        'seeds': [1244102535, 2063541227, 1096982489, 1916869805, 200476605]
    },
    'higgs': {
        'algos': ['SBSMU', 'MU'],
        'time_limit': 15 * 15 * 60 + 1,
        'loss_calc_intvl': 15 * 60,
        'dataset': 'higgs_5nn_5clusters.mat',
        'seeds': [1244102535]
    }
}

DATASET_DIR = Path('src/datasets')
RESULTS_DIR = Path('src/exp_benchmark')
START_TIME = default_timer()


def ts():
    """Timestamp for print statements"""
    t = default_timer() - START_TIME
    return f"[{datetime.timedelta(seconds=t)}]"


def main(exp_name):
    print(f"{ts()} Starting experiment '{exp_name}'", flush=True)
    exp = EXPERIMENTS[exp_name]
    results_path = RESULTS_DIR.joinpath(f'{exp_name}.pkl')
    dataset_path = DATASET_DIR.joinpath(exp['dataset'])
    X, rank = load_matrix(dataset_path)
    results = {}

    for alg_name in exp['algos']:
        alg_dict = ALGOS[alg_name]
        alg_func = alg_dict['algorithm']
        config = alg_dict['config']
        results[alg_name] = {}
        print(f"\n{ts()} Algorithm: {alg_name}", flush=True)

        for seed in exp['seeds']:
            print(f"\n{ts()} Seed: {seed}", flush=True)
            t0 = default_timer()
            W_copies, times, iter_counts = alg_func(X, rank, time_limit=exp['time_limit'], seed=seed,
                                                    loss_calc_intvl=exp['loss_calc_intvl'], **config)
            t1 = default_timer()
            print(f"{ts()} Completed factorization in {round(t1 - t0, 2)} s", flush=True)
            losses = [i_divergence(X, W) for W in W_copies]
            t2 = default_timer()
            print(f"{ts()} Completed loss calculations in {round(t2 - t1, 2)} s", flush=True)
            print(f"{ts()} Min loss: {min(losses)}", flush=True)
            print(f"{ts()} Iterations: {iter_counts[-1]}", flush=True)

            results[alg_name][seed] = {'losses': losses, 'times': times, 'iter_counts': iter_counts}

        with results_path.open('wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    mp.freeze_support()
    main(sys.argv[1])
