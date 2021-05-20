#!/bin/bash

module load Anaconda3/2020.07;

for datasetName in 20newsgroups curet amlall dolphins football gisette iris wine; do
  for ((i = 0 ; i < "$1" ; i++)); do
    sbatch --job-name="$datasetName" src/exp_hyperparams/run.py "$1" "$i" "$datasetName";
  done
done

