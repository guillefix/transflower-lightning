#!/bin/bash

#export TPU_IP_ADDRESS=10.8.195.90;
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
#export XRT_TPU_CONFIG="tpu_worker;0;"

py=python3
#py=python
#py='python3 -m torch_xla.distributed.xla_dist --tpu='${TPU_NAME}' --conda-env=torch-xla-nightly -- python'
dataset=multimodal
model=transflower
#exp=aistpp_big
exp=aistpp_flower_test2

#$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=32 --num_windows=1 --max_epochs=20000\
$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=4 --num_windows=1 --max_epochs=20000\
    --experiment_name=$exp\
    --lr_policy="multistep" \
    --lr_decay_milestones="[5000,10000]" \
    --learning_rate=2e-4 \
    --dins="219,103" \
    --douts="219" \
    --input_modalities="joint_angles_scaled,mel_ddcpca_scaled" \
    --output_modalities="joint_angles_scaled" \
    --input_lengths="120,240" \
    --output_lengths="1" \
    --output_time_offset="121" \
    --predicted_inputs="0,0" \
    --nlayers=10 \
    --nhead=10 \
    --num_glow_coupling_blocks=3 \
    --glow_use_attn \
    --use_transformer_nn \
    --use_pos_emb_coupling \
    --use_pos_emb_output \
    --dhid=800 \
    --glow_norm_layer="batchnorm" \
    --glow_bn_momentum=0.1 \
    --dropout=0.1 \
    --workers=$(nproc) \
    --gradient_clip_val=0.5 \
    --gpus=1 \
    --workers=0 \
#    --accelerator=ddp \
#    --workers=$(nproc) \
    #--continue_train \
    #--tpu_cores=8 \
#    --log_every_n_steps=1 \
#    --flush_logs_every_n_steps=1 \
    #--learning_rate=3e-5 \
