#!/bin/bash

export TPU_IP_ADDRESS=10.112.91.170;
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
#export XRT_TPU_CONFIG="tpu_worker;0;"

py=python3
#py=python
#py='python3 -m torch_xla.distributed.xla_dist --tpu='${TPU_NAME}' --conda-env=torch-xla-nightly -- python'
dataset=multimodal
model=moglow
#exp=aistpp_big
exp=moglow_loc_og_test

$py training/train.py --data_dir=data/moglow_loc --dataset_name=$dataset --model=$model --batch_size=128 --num_windows=1 --max_epochs=20000\
    --experiment_name=$exp\
    --lr_policy="multistep" \
    --network_model="LSTM" \
    --lr_decay_milestones="[5000,10000]" \
    --learning_rate=2e-4 \
    --dins="63,3" \
    --input_modalities="moglow_loc,moglow_loc_control" \
    --output_modalities="moglow_loc" \
    --input_lengths="79,80" \
    --output_lengths="70" \
    --output_time_offset="10" \
    --dropout=0 \
    --workers=$(nproc) \
    --gpus=1 \
#    --input_lengths="79,80" \
#    --output_lengths="70" \
#    --accelerator=ddp \
#    --continue_train \
#    --tpu_cores=8 \
#    --gradient_clip_val=0.5 \
#    --log_every_n_steps=1 \
#    --flush_logs_every_n_steps=1 \
    #--learning_rate=3e-5 \
#    --input_lengths="10,11" \
#    --output_lengths="10" \
#    --output_time_offset="1" \
#    --input_seq_lens="1,2" \
