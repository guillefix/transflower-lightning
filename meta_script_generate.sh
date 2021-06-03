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

#base_filenames_file=base_filenames_test.txt
base_filenames_file=base_filenames_test_test.txt
#base_filenames_file=base_filenames_test_test2.txt


#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos moglow_moglow_pos
#for exp in moglow_trans_moglow_pos
#for exp in transglower_moglow_pos transglower_residual_moglow_pos transflower_residual_moglow_pos transflower_moglow_pos residualflower2_transflower_moglow_pos moglow_trans_moglow_pos moglow_moglow_pos
#for exp in transglower_aistpp_expmap transglower_residual_aistpp_expmap transflower_residual_aistpp_expmap transflower_aistpp_expmap residualflower2_transflower_aistpp_expmap moglow_aistpp_expmap
#for exp in transflower_residual_aistpp_expmap transflower_aistpp_expmap_future1 transflower_aistpp_expmap_future3 residualflower2_transflower_aistpp_expmap_future1 residualflower2_transflower_aistpp_expmap_future3 moglow_aistpp_expmap 
#for exp in transflower_residual_aistpp_expmap_future3
#for exp in transflower_expmap_use_pos_emb_output moglow_expmap transflower_expmap_studentT_gclp1
#for exp in transflower_expmap_use_pos_emb_output
#for exp in transformer_expmap_no_pos_emb_output
#for exp in transformer_expmap transformer_expmap_no_pos_emb_output
#for exp in transflower_expmap moglow_expmap transflower_residual_expmap_1e4 transflower_expmap_large mowgli_expmap_stage2
#for exp in transflower_expmap moglow_expmap transflower_expmap_large mowgli_expmap_stage2 transflower_expmap_xt
#for exp in transflower_expmap moglow_expmap transflower_expmap_large transflower_expmap_xt
#for exp in transformer_expmap1
#for exp in transformer_expmap_no_pos_emb_output
#for exp in transflower_expmap_large_ext_nsn transflower_expmap_large_ext
#for exp in transflower_expmap_old
#for exp in transflower_expmap_finetune2_old
#for exp in transflower_residual_expmap
#for exp in transflower_expmap_large_ext_newdata
#for exp in transflower_expmap_newdata
#for exp in moglow_expmap
#for exp in moglow_expmap transflower_expmap mowgli_expmap_future3
#for exp in moglow_expmap_finetune transflower_expmap_finetune2 moglow_expmap transflower_expmap

#for exp in moglow_expmap_finetune transflower_expmap_finetune2 transflower_expmap_finetune2_old moglow_expmap transflower_expmap transflower_expmap_old transformer_expmap
#for exp in transflower_expmap transflower_expmap_old transformer_expmap moglow_expmap transflower_expmap_finetune2_old transflower_expmap_finetune2
#for exp in transflower_expmap_old transformer_expmap moglow_expmap transflower_expmap_finetune2_old transflower_expmap_finetune2
#for exp in transformer_expmap moglow_expmap transflower_expmap_finetune2_old
#for exp in transflower_expmap transformer_expmap
#for exp in transflower_expmap_finetune2_old moglow_expmap
#for exp in moglow_expmap
#for exp in transflower_expmap_finetune2 transflower_expmap_old
#for exp in transflower_expmap_newdata
#for exp in transflower_expmap_large_newdata mowgli_expmap_stage2_newdata3
#for exp in transflower_expmap_large_newdata
for exp in transflower_expmap_large_cr_newdata
#for exp in transflower_expmap_cr4_label7_newdata
#for exp in transflower_expmap_large_cr_newdata_nomirror
#for exp in transflower_expmap_smoldata
#for exp in mowgli_expmap_stage2_newdata3
#for exp in transflower_expmap

#for exp in transformer_expmap
#for exp in mowgli_expmap_stage2_newdata
#for exp in transflower_expmap
#for exp in transflower_expmap_finetune2 transformer_expmap1
#for exp in transformer_expmap1
#for exp in transflower_expmap_newdata
#for exp in transformer_expmap1
#for exp in transflower_expmap2
#for exp in mowgli_expmap_future10_fix transflower_residual_expmap
do
	while read line; do
		  echo "$line"
		#sbatch slurm_script_generate.slurm $exp $line
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test_original_seeds
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test1
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test1
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test1 --seeds expmap_scaled_20,the_basement
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test1 --seeds_file expmap_scaled_20,the_basement
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined2_test
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined2_test --seeds expmap_scaled_20,kthmisc_12
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined2_test --seeds expmap_cr_scaled_20,kthmisc_12
		sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined2_test --seeds expmap_cr_scaled_20,kthmisc_12 --generate_video
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3_test --seeds expmap_cr_scaled_20,aistpp_gHO_sBM_cAll_d19_mHO3_ch10;dance_style,aistpp_gHO_sBM_cAll_d19_mHO3_ch10
		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3 --seeds "expmap_cr_scaled_20,aistpp_gHO_sBM_cAll_d19_mHO3_ch10;dance_style,aistpp_gHO_sBM_cAll_d19_mHO3_ch10" --generate_video

		#for i in 1 2 3 4 5; do
		#	mkdir inference/generated_${i}/
		#	mkdir inference/generated_${i}/${exp}
		#	mkdir inference/generated_${i}/${exp}/predicted_mods
		#	mkdir inference/generated_${i}/${exp}/videos
		#	#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i}
		#	sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i} --seeds_file $SCRATCH/data/seeds_${i}
		#done
	done <$base_filenames_file

	#for i in 1 2 3 4 5; do
	#for i in 5; do
	#	mkdir inference/generated_${i}/
	#	mkdir inference/generated_${i}/${exp}
	#	mkdir inference/generated_${i}/${exp}/predicted_mods
	#	mkdir inference/generated_${i}/${exp}/videos
	#	#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i}
	#	#sbatch slurm_script_generate_array.slurm $exp $i
	#	sbatch slurm_script_generate_array_rs.slurm $exp $i
	#done

	#sbatch slurm_script_generate.slurm $exp aistpp_gMH_sFM_cAll_d22_mMH3_ch04 --generate_bvh --data_dir=${SCRATCH}/data/dance_combined2
	#sbatch slurm_script_generate.slurm $exp fan
	#sbatch slurm_script_generate.slurm $exp polish_cow
	#sbatch slurm_script_generate.slurm $exp aistpp_gLO_sBM_cAll_d14_mLO4_ch02
done

#for exp in transflower_residual_aistpp_expmap_future1 transflower_aistpp_expmap_future1 residualflower2_transflower_aistpp_expmap_future1_future1
#do
#	sbatch slurm_script_generate.slurm $exp
#done
