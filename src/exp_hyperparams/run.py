#!/cluster/apps/eb/software/Anaconda3/2020.07/bin/python
#SBATCH --account=share-ie-idi
#SBATCH --time=96:00:00
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --mem=24000
"""Hyperparameter experiment"""
import os
import sys
from itertools import product
import multiprocessing as mp
from pathlib import Path
import pickle
from timeit import default_timer
import datetime

import numpy as np

sys.path.append(os.getcwd())


from src.SBSMU import SBSMU
from src.helpers import load_matrix

DATASET_DIR = Path('src/datasets/')
RESULTS_DIR = Path('src/exp_hyperparams/results/')
SEEDS = [1573942128, 274259859, 1978143047, 1059581820, 923533011,
         1243444734, 976315392, 1283209697, 347312645, 403974349]
ALPHAS = [0.9, 0.99, 0.999, 0.9999, 0.99999]
BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]
ETAS = [0.1, 0.3, 0.5, 0.7, 0.9]
START_TIME = default_timer()

EXPERIMENTS = {
    '20newsgroups': {
        'filepath': DATASET_DIR.joinpath('20newsgroups_10k_5nn.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'curet': {
        'filepath': DATASET_DIR.joinpath('curet_10nn.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'amlall': {
        'filepath': DATASET_DIR.joinpath('amlall_5nn.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'dolphins': {
        'filepath': DATASET_DIR.joinpath('dolphins.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'football': {
        'filepath': DATASET_DIR.joinpath('football.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'gisette': {
        'filepath': DATASET_DIR.joinpath('gisette_10nn.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'iris': {
        'filepath': DATASET_DIR.joinpath('iris_5nn.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
    'wine': {
        'filepath': DATASET_DIR.joinpath('UCI_wine_5nn.mat'),
        'time_limit': 600,
        'loss_calc_intvl': 0.1
    },
}


def ts():
    """Timestamp for print statements"""
    t = default_timer() - START_TIME
    return f"[{datetime.timedelta(seconds=t)}]"


def get_configs(n_workers, worker_id, results_dir):
    """Generate the list of configurations to be tested by this worker"""
    np.random.seed(SEEDS[0])
    all_configs = list(product(ALPHAS, BETAS, ETAS))

    # Determine which configs have already been run
    run_configs = []
    for file_path in results_dir.iterdir():
        with open(file_path, 'rb') as handle:
            results = pickle.load(handle)
            run_configs.extend(results.keys())
    all_configs = [c for c in all_configs if c not in run_configs]

    np.random.shuffle(all_configs)
    config_chunks = np.array_split(all_configs, n_workers)
    return config_chunks[worker_id]


def main(n_workers, worker_id, dataset_name):
    print(f"\n{ts()} Dataset: {dataset_name}")
    results_dir = RESULTS_DIR.joinpath(dataset_name)
    results_dir.mkdir(exist_ok=True, parents=True)
    dataset_dict = EXPERIMENTS[dataset_name]
    dataset_path = dataset_dict['filepath']
    X, rank = load_matrix(dataset_path)
    time_limit = dataset_dict['time_limit']
    loss_calc_intvl = dataset_dict['loss_calc_intvl']

    results_path = results_dir.joinpath(f'{worker_id}.pkl')
    if results_path.is_file():
        with open(results_path, 'rb') as handle:
            results = pickle.load(handle)
        print(f"{ts()} Results file found and loaded.")
    else:
        results = {}
        print(f"{ts()} Results file not found. New file created.")

    configs = get_configs(n_workers, worker_id, results_dir)
    print(f"{ts()} Configs: \n {configs}", flush=True)

    for alpha, beta, eta in configs:
        print(f"\n{ts()} Config: ({alpha}, {beta}, {eta})", flush=True)
        runs = {}
        for seed in SEEDS:
            print(f"\n{ts()} Seed: {seed}", flush=True)
            losses, times, iter_counts = SBSMU(X, rank, eta=eta, alpha=alpha, beta=beta,
                                               time_limit=time_limit, loss_calc_intvl=loss_calc_intvl, seed=seed)
            print(f"{ts()} Min loss: {min(losses)}, iterations: {iter_counts[-1]}", flush=True)
            runs[seed] = {'losses': losses, 'times': times, 'iter_counts': iter_counts}

        results[(alpha, beta, eta)] = runs

        with results_path.open('wb') as f:
            pickle.dump(results, f)

    print(f"{ts()} Completed testing for all scheduled configurations", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
