#!/bin/bash

#SBATCH --time=1:00:00
##SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_gpu-dev
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH -A imi@gpu
##SBATCH -C v100-32g
##SBATCH --array=transflower_expmap,transflower_expmap_old,transformer_expmap,moglow_expmap,transflower_expmap_finetune2_old,transflower_expmap_finetune2%100
#SBATCH --array=0-278%100

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
#./script_generate.sh $@
exp=$1
i=$2
export SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID+1))
line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" base_filenames_test.txt)
./script_generate.sh $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i}

