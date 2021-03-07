#!/bin/bash

folder=$1
py=python
# n=$(nproc)
n=1

#find $1 -type f -iname "*.mp3" -exec basename \{\} .mp3 \; > $1/base_filenames.txt

#mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name multi_mel --feature_size 80 --step_size 0.01 --notranspose
#mpirun -n $n $py ./scripts/feature_extraction/generate_ddc_features.py $1 --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir scripts/training
#mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name mel --feature_size 100
#mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name envelope --notranspose
#mpirun -n 1 $py ./scripts/feature_extraction/extract_transform.py $1 --feature_name mp3_multi_mel_80.npy_ddc_hidden --transforms pca_transform
#mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name mp3_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
#mpirun -n $n $py ./scripts/feature_extraction/combine_feats.py $1 $1/base_filenames.txt --feature_names mp3_mel_100,mp3_envelope_100,ddcpca --new_feature_name mel_ddcpca
#mpirun -n 1 $py ./scripts/feature_extraction/extract_transform2.py $1 --feature_name mel_ddcpca --transforms scaler
#mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name mel_ddcpca --transform_name scaler --new_feature_name mel_ddcpca_scaled

mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name multi_mel --feature_size 80 --step_size 0.01 --notranspose
mpirun -n $n $py ./scripts/feature_extraction/generate_ddc_features.py $1 --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir scripts/training
mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name mel --feature_size 100
mpirun -n $n $py ./scripts/feature_extraction/process_audio.py $1 --feature_name envelope --notranspose
mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name mp3_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
mpirun -n $n $py ./scripts/feature_extraction/combine_feats.py $1 $1/base_filenames.txt --feature_names mp3_mel_100,mp3_envelope_100,ddcpca --new_feature_name mel_ddcpca
mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name mel_ddcpca --transform_name scaler --new_feature_name mel_ddcpca_scaled
