#!/bin/bash

#SBATCH --time=1:00:00
##SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_gpu-dev
##SBATCH --ntasks=1 --cpus-per-task=1 --gres=gpu:1
#SBATCH --ntasks=1 --cpus-per-task=4
##SBATCH -A imi@gpu
#SBATCH -A imi@cpu
##SBATCH -C v100-32g

#export PYTHONPATH=$SCRATCH/:${PYTHONPATH}

module load pytorch-gpu/py3/1.8.0

#exp=$1
#seq_id=$2
#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos

#./script_train_residualflower.sh
./feature_extraction/audio_feature_extraction_test.sh $@

