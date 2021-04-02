#!/bin/bash

#if using XLA
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

#py=python3
py=python3
#exp=aistpp_2
#exp=aistpp_flower_1
#exp=aistpp_flower_gpu_nodrop
exp=moglow_loc
seq_id="1"

mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/predicted_mods
mkdir inference/generated/${exp}/videos

$py inference/generate.py --data_dir=test_data/moglow_loc_test --experiment_name=$exp \
    --seq_id $seq_id \
    --input_modalities="moglow_loc,moglow_loc_control" \
    --output_modalities="moglow_loc" \

#$py analysis/aistplusplus_api/generate_video_from_mats.py --pred_mats_file inference/generated/${exp}/predicted_mods/${seq_id}.pkl_joint_angles_mats.generated.npy \
#    --output_folder inference/generated/${exp}/videos/ \
#    --audio_file test_data/${seq_id}.mp3 \
#    --trim_audio 2


