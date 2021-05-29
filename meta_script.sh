#!/bin/bash

#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos

#exp=transglower_aistpp_expmap
#exp=transglower_residual_aistpp_expmap
#exp=transflower_residual_aistpp_expmap
#exp=transflower_aistpp_expmap
#exp=residualflower2_transflower_aistpp_expmap
#exp=moglow_aistpp_expmap


#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos moglow_moglow_pos
#for exp in moglow_trans_moglow_pos
#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos moglow_trans_moglow_pos moglow_moglow_pos
#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos
#for exp in moglow_trans_moglow_pos
#for exp in transglower_aistpp_expmap transglower_residual_aistpp_expmap transflower_residual_aistpp_expmap transflower_aistpp_expmap residualflower2_transflower_aistpp_expmap moglow_aistpp_expmap
#for exp in transglower_aistpp_expmap transglower_residual_aistpp_expmap
#for exp in transflower_residual_aistpp_expmap transflower_aistpp_expmap
#for exp in moglow_expmap transformer_expmap
#for exp in transflower_expmap
#for exp in moglow_expmap
#for exp in transflower_expmap_large2
#for exp in transformer_expmap_large
#for exp in transflower_expmap_large
#for exp in transflower_expmap_old
#for exp in mowgli_expmap_stage2
#for exp in mowgli_expmap_bigcode
#for exp in transflower_expmap_large_cr2
#for exp in transflower_expmap_large_cr
for exp in transflower_expmap_cr4
#for exp in mowgli_expmap
#for exp in mowgli_expmap_nocond
#for exp in mowgli_expmap transflower_residual_expmap
#for exp in transformer_residual_expmap
#for exp in transflower_expmap
#for exp in transflower_expmap_large

#for exp in transflower_residual_expmap
#for exp in transflower_expmap_large_ext
#for exp in mowgli_expmap_future3
#for exp in transflower_residual_aistpp_expmap
#for exp in moglow_expmap transflower_expmap
do
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata3 --num_nodes 8 --continue_train --no_load_hparams 
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata3 --num_nodes 8 --continue_train
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata4 --num_nodes 8 --continue_train --no_load_hparams 
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 8 --continue_train


	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata_posemb --num_nodes 1  --continue_train
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_posemb --num_nodes 1 --data_dir=${SCRATCH}/data/dance_combined --continue_train
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --hparams_file=training/hparams/dance_combined/moglow_expmap2.yaml
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train
	sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1

	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train --max_epochs=100
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata_nomirror --num_nodes 1 --base_filenames_file base_filenames_train_nomirror.txt

	#sbatch slurm_script3.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train --max_epochs=10000

	#sbatch slurm_script_dev.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --fix_lengths
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --no_load_hparams
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata_nocond
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata $@ --num_nodes 8
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata $@ --num_nodes 1 --continue_train
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata $@ --num_nodes 4
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata2 --max_prior_loss_weight=0 --num_nodes 8
	#sbatch slurm_script_dev.slurm $exp --experiment_name testing $@
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata_mix --num_mixture_components=3 --learning_rate=3e-5 $@
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_finetune2 --base_filenames_file base_filenames_train_finetune.txt --continue_train
	#sbatch slurm_script2.slurm $exp --continue_train --data_dir=${SCRATCH}/data/dance_combined
	#sbatch slurm_script2.slurm $exp --experiment_name=${exp}_smoldata --data_dir=${SCRATCH}/data/dance_combined --base_filenames_file base_filenames_train_finetune.txt
done

