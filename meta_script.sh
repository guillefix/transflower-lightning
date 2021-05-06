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
#for exp in moglow_expmap
#for exp in transformer_expmap
#for exp in mowgli_expmap
#for exp in mowgli_expmap transflower_residual_expmap
#for exp in transformer_residual_expmap
#for exp in transflower_expmap
#for exp in transflower_expmap_large
#for exp in transflower_residual_expmap
#for exp in transflower_expmap_large_ext
#for exp in mowgli_expmap_future3
#for exp in transflower_residual_aistpp_expmap
for exp in moglow_expmap transflower_expmap
do
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata --num_nodes 4
	sbatch slurm_script2.slurm $exp --experiment_name ${exp}_finetune --base_filenames_file base_filenames_train_finetune.txt --continue_train
done

