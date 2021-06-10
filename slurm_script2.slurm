#!/bin/bash

#SBATCH --time=100:00:00
##SBATCH --time=20:00:00
##SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_gpu-t3
##SBATCH --qos=qos_gpu-dev
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:4
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:1
##SBATCH --nodes=4 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
##SBATCH --nodes=2 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
##SBATCH --nodes=1 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=6 --gres=gpu:2
##SBATCH --partition=gpu_p2
#SBATCH -A imi@gpu
#SBATCH -C v100-32g
#SBATCH --exclusive

export MASTER_PORT=1234
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $slurm_nodes
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)
echo $MASTER_ADDRESS

module load pytorch-gpu/py3/1.8.0

#exp=$1
srun ./script_train.sh $@
#srun ./script_train_dev.sh $exp
