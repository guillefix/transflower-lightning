#!/bin/bash

#for exp in moglow_expmap1
#for exp in transflower_expmap_cr4_label_bs5
for exp in transflower_expmap_cr_label2
do
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata4 --num_nodes 8 --continue_train --no_load_hparams 
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 8 --continue_train

	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_posemb --num_nodes 1 --data_dir=${SCRATCH}/data/dance_combined --continue_train
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --hparams_file=training/hparams/dance_combined/moglow_expmap2.yaml
	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train --data_dir=${SCRATCH}/data/dance_combined2

	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1
	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train 
	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train --base_filenames_file base_filenames_train_nojd.txt
	sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 #--fix_lengths
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1

	#sbatch slurm_script2.slurm $exp --experiment_name=${exp}_smoldata --data_dir=${SCRATCH}/data/dance_combined --base_filenames_file base_filenames_train_finetune.txt
done

