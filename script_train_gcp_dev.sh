#!/bin/bash

export TPU_IP_ADDRESS=10.104.22.146;
#export TPU_IP_ADDRESS=10.95.66.34;
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
#export XRT_WORKERS="localservice:0;grpc://localhost:40934"
#export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"
#export PYTHONPATH=$SCRATCH/:${PYTHONPATH}
#export PYTHONPATH=/gpfsscratch/rech/imi/usc19dv/lib/python3.7/site-packages:${PYTHONPATH}

py=python3

#root_dir=$SCRATCH/data
root_dir=data

####aistpp_60hz
#data_dir=${root_dir}/scaled_features
#hparams_file=aistpp_60hz/transflower_aistpp_expmap
#hparams_file=aistpp_60hz/transglower_aistpp_expmap

####aistpp_20hz
data_dir=${root_dir}/aistpp_20hz
#exp=$1
exp=testing
#exp=transglower_aistpp_expmap
#exp=transglower_residual_aistpp_expmap
#exp=transflower_residual_aistpp_expmap
#exp=transflower_aistpp_expmap
#exp=residualflower2_transflower_aistpp_expmap
#exp=moglow_aistpp_expmap
#hparams_file=aistpp_20hz/${exp}
hparams_file=aistpp_20hz/mowgli_aistpp_expmap_testing

## Fix: needs vmapped version of transformer:
#hparams_file=aistpp_20hz/residualflower2_moglow_aistpp_expmap

####dance_combined
#data_dir=${root_dir}/dance_combined
#exp=$1
#exp=transflower_expmap
#exp=moglow_expmap
#hparams_file=dance_combined/${exp}

#exp=${exp}_future3_actnorm
#exp=${exp}_future3
#exp=${exp}_future3

echo $exp

$py training/train.py --data_dir=${data_dir} --max_epochs=1000\
    --model=mowgli2 \
    --do_validation \
    --val_batch_size=32 \
    --batch_size=32 \
    --experiment_name=$exp\
    --workers=$(nproc) \
    --tpu_cores=8 \
    --hparams_file=training/hparams/${hparams_file}.yaml \
    #--continue_train \
    #--load_weights_only \
    #--stage2 \
    #--prior_use_x_transformers \
    #--output_lengths="3" \
    #--max_prior_loss_weight=0.01 \
    #--accelerator=ddp \
    #--scales="[[16,0]]" \
#    --use_rotary_pos_emb \
    #--residual_scales="[[16,0]]"
#    --glow_norm_layer="actnorm" \
    #--use_pos_emb_output \
    #--gpus=2 \
