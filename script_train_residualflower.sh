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
model=residualflower2
#exp=aistpp_big
exp=aistpp_residual
exp=aistpp_residual2

#$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=32 --num_windows=1 --max_epochs=20000\
$py training/train.py --data_dir=data/scaled_features --dataset_name=$dataset --model=$model --batch_size=8 --num_windows=1 --max_epochs=20000\
    --experiment_name=$exp\
    --mean_model=transformer \
    --residual_model=moglow \
    --lr_policy="multistep" \
    --lr_decay_milestones="[25,50]" \
    --learning_rate=5e-5 \
    --dins="72,103" \
    --douts="72" \
    --input_modalities="expmap_scaled,mel_ddcpca_scaled" \
    --output_modalities="expmap_scaled" \
    --input_lengths="59,60" \
    --output_lengths="20" \
    --output_time_offset="40" \
    --mean_nlayers=6 \
    --mean_nhead=10 \
    --mean_use_pos_emb_output \
    --mean_dhid=800 \
    --dropout=0 \
    --workers=$(nproc) \
    --gpus=1 \
    --gradient_clip_val=0.5 \
    --residual_input_seq_lens="40,41" \
    --residual_glow_K=16 \
#    --residual_dhid=800 \
#    --residual_cond_concat_dims \
#    --residual_glow_norm_layer="batchnorm" \
#    --residual_glow_bn_momentum=0.1 \
#    --residual_scales="[[4,0], [4,0]]" \
#    --residual_num_glow_coupling_blocks=2 \
#    --residual_glow_use_attn \
#    --residual_use_transformer_nn \
#    --residual_use_pos_emb_coupling \
#    --residual_use_pos_emb_output \
    #--continue_train \
#    --load_weights_only \
#    --tpu_cores=8 \
#    --accelerator=ddp \
#    --workers=$(nproc) \
    #--continue_train \
#    --log_every_n_steps=1 \
#    --flush_logs_every_n_steps=1 \
    #--learning_rate=3e-5 \
