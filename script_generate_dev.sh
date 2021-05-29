#!/bin/bash

#if using XLA
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

py=python3

#exp=transglower_residual_aistpp_expmap
#exp=moglow_aistpp_expmap
#exp=testing
exp=transflower_expmap_old
#exp=mowgli_expmap_stage2_newdata
#exp=$1
#exp=mowgli_aistpp_expmap_future3
#exp=aistpp_residual
#seq_id=gKR_sFM_cAll_d28_mKR5_ch06
#seq_id=gLH_sFM_cAll_d16_mLH3_ch04
#seq_id=gPO_sFM_cAll_d12_mPO4_ch19
#seq_id=aistpp_gMH_sFM_cAll_d22_mMH3_ch04
seq_id=groovenet_2
echo $exp $seq_id

mkdir inference/generated/
mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/predicted_mods
mkdir inference/generated/${exp}/videos
fps=20
#fps=60
#data_dir=data/aistpp_20hz
data_dir=data/dance_combined
#data_dir=data/aistpp_60hz

# if we don't pass seq_id it will choose a random one from the test set
$py inference/generate.py --data_dir=$data_dir --output_folder=inference/generated --experiment_name=$exp \
    --generate_video \
    --seq_id $seq_id \
    --max_length 300


