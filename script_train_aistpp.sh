#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#export TPU_IP_ADDRESS=10.29.7.114;
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
#export XRT_WORKERS="localservice:0;grpc://localhost:40934"
#export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"


py=python3
#py=python
#py='python3 -m torch_xla.distributed.xla_dist --tpu='${TPU_NAME}' --conda-env=torch-xla-nightly -- python'
dataset=multimodal
model=transformer
#exp=aistpp_big
exp=aistpp_test

$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=8 --num_windows=1 --max_epochs=20000 \
    --experiment_name=$exp\
    --optimizer=adam \
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
    --workers=$(nproc) \
    --num_processes=$(nproc) \
    --nhead=10 \
    --dhid=800 \
    --dropout=0 \
    --use_pos_emb_output \
    --gpus=1 \
    --accelerator=ddp \
    --gradient_clip_val=0.5 \
#    --continue_train \
    #--tpu_cores=8 \
