#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6

#find $1 -type f -iname "*.mp3" -exec basename \{\} .mp3 \; > $1/base_filenames.txt

fps=20
format=wav

###SEQUENCE TO PROCESS DATA WHEN NEEDING TO COMPUTE NORMALIZATION TRANSFORMS
#mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names multi_mel --mel_feature_size 80 --fps 100 # fps=100 coz thats what ddc expects
#mpirun -n $n $py ./feature_extraction/generate_ddc_features.py $@ --audio_format $format --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir feature_extraction --fps $fps
#mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names mel,envelope,madmombeats --mel_feature_size 80 --fps $fps --combined_feature_name audio_feats
#mpirun -n 1 $py ./feature_extraction/extract_transform.py $1 --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transforms pca_transform
#mpirun -n $n $py ./feature_extraction/apply_transforms.py $@ --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
#./feature_extraction/script_to_list_filenames $1 $format
#mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names ${format}_audio_feats,ddcpca --new_feature_name feats_ddcpca
#mpirun -n 1 $py ./feature_extraction/extract_transform2.py $1 --feature_name feats_ddcpca --transforms scaler
#mpirun -n $n $py ./feature_extraction/apply_transforms.py $@ --feature_name feats_ddcpca --transform_name scaler --new_feature_name audio_feats_scaled_${fps}

###SEQUENCE WHEN USING EXISTING TRANSFORMS (SO NO NEED T RECOMPUTE THEM)
#NEED TO UPDATE
#mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name multi_mel --feature_size 80 --step_size 0.01 --notranspose
#mpirun -n $n $py ./scripts/feature_extraction/generate_ddc_features.py $1 --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir training/experiments
#mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name mel --feature_size 20
#mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name envelope --notranspose
#mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
#mpirun -n $n $py ./scripts/feature_extraction/combine_feats.py $1 $1/base_filenames.txt --feature_names ${format}_mel_100,mp3_envelope_100,ddcpca --new_feature_name mel_ddcpca
#mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name mel_ddcpca --transform_name scaler --new_feature_name mel_ddcpca_scaled
