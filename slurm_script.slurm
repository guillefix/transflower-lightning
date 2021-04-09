#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --ntasks=1 --cpus-per-task=16 --gres=gpu:4
#SBATCH -A imi@gpu

./script_train_residualflower.sh

