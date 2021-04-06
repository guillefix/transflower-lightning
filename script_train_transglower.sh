#!/bin/bash

#export TPU_IP_ADDRESS=10.8.195.90;
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

py=python3
#py=python
#py='python3 -m torch_xla.distributed.xla_dist --tpu='${TPU_NAME}' --conda-env=torch-xla-nightly -- python'
dataset=multimodal
model=transglower
#exp=aistpp_big
exp=aistpp_transglower

#$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=32 --num_windows=1 --max_epochs=20000\
$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=8 --num_windows=1 --max_epochs=20000\
    --experiment_name=$exp\
    --lr_policy="multistep" \
    --lr_decay_milestones="[5000,10000]" \
    --learning_rate=1e-4 \
    --dins="219,103" \
    --douts="219" \
    --input_modalities="joint_angles_scaled,mel_ddcpca_scaled" \
    --output_modalities="joint_angles_scaled" \
    --input_lengths="29,30" \
    --input_seq_lens="10,11" \
    --output_lengths="20" \
    --output_time_offset="10" \
    --nlayers=4 \
    --nhead=10 \
    --use_pos_emb_output \
    --dhid=800 \
    --glow_K=16 \
    --dropout=0 \
    --workers=$(nproc) \
    --gpus=1 \
#    --continue_train \
#    --tpu_cores=8 \
#    --gradient_clip_val=0.5 \
#    --accelerator=ddp \
#    --workers=$(nproc) \
    #--continue_train \
#    --log_every_n_steps=1 \
#    --flush_logs_every_n_steps=1 \
    #--learning_rate=3e-5 \