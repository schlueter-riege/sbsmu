# SBSMU
A novel algorithm using stochastic multiplicative updates for symmetric nonnegative matrix factorization.

## Requirements
The project has been developed in Python 3, using the interpreter and libraries that come bundled with Anaconda 2020.07. The required packages are also listed in ```requirements.txt```. The hyperparameter experiment is designed to run in a cluster environment using SLURM.

## Usage
To run the hyperparameter experiment in a cluster enviroment:

```src/exp_hyperparams/run.sh <workers per dataset> ```

To run the hyperparameter experiment on a local computer:

```src/exp_hyperparams/run.py 1 1 <dataset> ```

To run the MNIST or Higgs experiment:

```src/exp_hyperparams/run.py <mnist | higgs> ```

For more details, see the documentation in the Python files.

## Repository Structure
```
.
├── src                        
│   ├── datasets               # Processed datasets, not uploaded to Github
│   ├── exp_benchmark          # Benchmark experiments on MNIST and Higgs
│   │   ├── run.py             # Experiment setup
│   │   └── visualization.py   # Visualization of results
│   ├── exp_hyperparams        # Hyperparameter experiment
│   │   ├── run.py             # Experiment setup
│   │   ├── run.sh             # Script for running experiment on cluster
│   │   └── visualization.py   # Visualization of results
│   ├── EGD.py                 # Exponentiated gradient descent implementation
│   ├── helpers.py             # Code shared by algorithms
│   ├── MU.py                  # Full-batch multiplicative updates implementation
│   ├── PSGD.py                # Projected stochastic gradient descent implementation
│   └── SBSMU.py               # Stochastic bound-and-scale multiplicative updates implementation
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt             # Python package requirements
```