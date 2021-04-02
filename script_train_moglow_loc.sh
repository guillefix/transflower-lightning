#!/bin/bash

export TPU_IP_ADDRESS=10.112.91.170;
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
#export XRT_TPU_CONFIG="tpu_worker;0;"

#py=python3
py=python
#py='python3 -m torch_xla.distributed.xla_dist --tpu='${TPU_NAME}' --conda-env=torch-xla-nightly -- python'
dataset=multimodal
model=transflower
#exp=aistpp_big
exp=moglow_loc

$py training/train.py --data_dir=data/moglow_loc --dataset_name=$dataset --model=$model --batch_size=128 --num_windows=1 --max_epochs=20000\
    --experiment_name=$exp\
    --lr_policy="multistep" \
    --lr_decay_milestones="[5000,10000]" \
    --learning_rate=2e-4 \
    --dins="63,3" \
    --douts="63" \
    --input_modalities="moglow_loc,moglow_loc_control" \
    --output_modalities="moglow_loc" \
    --input_lengths="10,10" \
    --output_lengths="1" \
    --output_time_offset="10" \
    --predicted_inputs="0,0" \
    --nlayers=6 \
    --nhead=10 \
    --num_glow_coupling_blocks=5 \
    --glow_use_attn \
    --use_transformer_nn \
    --use_pos_emb_coupling \
    --use_pos_emb_output \
    --dhid=800 \
    --dropout=0 \
    --workers=$(nproc) \
    --tpu_cores=8 \
    --continue_train \
#    --gradient_clip_val=0.5 \
#    --accelerator=ddp \
#    --gpus=0 \
#    --log_every_n_steps=1 \
#    --flush_logs_every_n_steps=1 \
    #--learning_rate=3e-5 \
