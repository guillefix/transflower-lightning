#!/bin/bash

#SBATCH --time=100:00:00
##SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:4
##SBATCH --partition=gpu_p2
#SBATCH -A imi@gpu
#SBATCH -C v100-32g
##SBATCH --exclusive

#export PYTHONPATH=$SCRATCH/:${PYTHONPATH}

module load pytorch-gpu/py3/1.8.0

exp=$1
#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos

#./script_train_residualflower.sh
#./script_train.sh $exp
./script_train2.sh $exp
#./script_train3.sh $exp

