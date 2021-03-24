#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

export TPU_NAME=tpu-quickstart

py=python3
#py=python
#py='python3 -m torch_xla.distributed.xla_dist --tpu='${TPU_NAME}' --conda-env=torch-xla-nightly -- python'
dataset=multimodal
model=transformer
#exp=aistpp_big
exp=aistpp_test

$py /home/guillefix/mt-lightning/training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=32 --num_windows=1 --max_epochs=500 \
    --experiment_name=$exp\
    --tpu_cores=8 \
    --learning_rate=3e-5 \
    --dins="219,103" \
    --douts="219" \
    --input_modalities="joint_angles_scaled,mel_ddcpca_scaled" \
    --output_modalities="joint_angles_scaled" \
    --input_lengths="120,240" \
    --output_lengths="20" \
    --output_time_offset="121" \
    --predicted_inputs="0,0" \
    --nlayers=12 \
    --num_processes=$(nproc) \
    --nhead=10 \
    --dhid=800 \
    --dropout=0 \
    #--continue_train \
