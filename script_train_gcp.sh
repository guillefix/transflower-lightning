#!/bin/bash

#export TPU_IP_ADDRESS=10.104.22.146;
#export TPU_IP_ADDRESS=10.95.66.34;
#export TPU_IP_ADDRESS=10.65.226.162;
export TPU_IP_ADDRESS=10.21.219.242;
#export TPU_IP_ADDRESS=10.93.151.138;
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
#data_dir=${root_dir}/aistpp_20hz
##exp=$1
#exp=mowgli_aistpp_expmap
##exp=transglower_aistpp_expmap
##exp=transglower_residual_aistpp_expmap
##exp=transflower_residual_aistpp_expmap
##exp=transflower_aistpp_expmap
##exp=residualflower2_transflower_aistpp_expmap
##exp=moglow_aistpp_expmap
#hparams_file=aistpp_20hz/${exp}

## Fix: needs vmapped version of transformer:
#hparams_file=aistpp_20hz/residualflower2_moglow_aistpp_expmap

####dance_combined
#data_dir=${root_dir}/dance_combined
data_dir=${root_dir}/dance_combined2
#exp=$1
#exp=transflower_expmap_large
#exp=transflower_residual_expmap
exp=transflower_expmap_cr
#exp=transformer_expmap
#exp=moglow_expmap
hparams_file=dance_combined/${exp}

#exp=${exp}_future3_actnorm
#exp=${exp}_future3
#exp=${exp}_future3
#exp=${exp}_no_pos_emb_output
#exp=${exp}_studentt
#exp=${exp}_large

echo $exp

$py training/train.py --data_dir=${data_dir} --max_epochs=300\
    --batch_size=42 \
    --hparams_file=training/hparams/${hparams_file}.yaml \
    --experiment_name=$exp\
    --workers=$(nproc) \
    --tpu_cores=8 \
    --continue_train \
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
