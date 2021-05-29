#!/bin/bash

#export TPU_IP_ADDRESS=10.104.22.146;
#export TPU_IP_ADDRESS=10.95.66.34;
#export TPU_IP_ADDRESS=10.65.226.162;
#export TPU_IP_ADDRESS=10.21.219.242;
#export TPU_IP_ADDRESS=10.93.151.138;
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

py=python3

root_dir=data

data_dir=${root_dir}/dance_combined2
#exp=transflower_expmap_cr2
exp=transglower_expmap_cr
hparams_file=dance_combined/${exp}


echo $exp

$py training/train.py --data_dir=${data_dir} --max_epochs=300\
    --hparams_file=training/hparams/${hparams_file}.yaml \
    --experiment_name=$exp\
    --tpu_cores=8 \
    --workers=$(nproc) \
    #--continue_train \
    #--sync_batchnorm \
    #--optimizer=madgrad \
    #--learning_rate=1e-3 \
    #--batch_size=128 \
    #--use_x_transformers \
    #--use_rotary_pos_emb \
    #--accelerator=ddp \
    #--flow_dist=studentT \
    #--no-use_pos_emb_output \
    #--load_weights_only \
    #--stage2 \
    #--prior_use_x_transformers \
    #--output_lengths="3" \
    #--max_prior_loss_weight=0.01 \
    #--scales="[[16,0]]" \
    #--residual_scales="[[16,0]]"
#    --glow_norm_layer="actnorm" \
    #--use_pos_emb_output \
    #--gpus=2 \
