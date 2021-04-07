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
model=residualflower
#exp=aistpp_big
exp=aistpp_residual

#$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=32 --num_windows=1 --max_epochs=20000\
$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=64 --num_windows=1 --max_epochs=20000\
    --fix_lengths \
    --experiment_name=$exp\
    --lr_policy="multistep" \
    --lr_decay_milestones="[25,50]" \
    --learning_rate=5e-5 \
    --dins="72,103" \
    --douts="72" \
    --input_modalities="expmap_scaled,mel_ddcpca_scaled" \
    --output_modalities="expmap_scaled" \
    --input_lengths="60,120" \
    --output_lengths="10" \
    --output_time_offset="60" \
    --predicted_inputs="0,0" \
    --nlayers=6 \
    --nhead=10 \
    --scales="[[4,0], [4,0]]" \
    --num_glow_coupling_blocks=2 \
    --glow_use_attn \
    --use_transformer_nn \
    --use_pos_emb_coupling \
    --use_pos_emb_output \
    --dhid=800 \
    --cond_concat_dims \
    --glow_norm_layer="batchnorm" \
    --glow_bn_momentum=0.1 \
    --dropout=0 \
    --workers=$(nproc) \
    --gpus=1 \
    --gradient_clip_val=0.5 \
    #--continue_train \
#    --load_weights_only \
#    --tpu_cores=8 \
#    --accelerator=ddp \
#    --workers=$(nproc) \
    #--continue_train \
#    --log_every_n_steps=1 \
#    --flush_logs_every_n_steps=1 \
    #--learning_rate=3e-5 \
