#!/bin/bash

#export TPU_IP_ADDRESS=10.8.195.90;
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"
#export PYTHONPATH=$SCRATCH/:${PYTHONPATH}
#export PYTHONPATH=/gpfsscratch/rech/imi/usc19dv/lib/python3.7/site-packages:${PYTHONPATH}
module load pytorch-gpu/py3/1.8.0

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
#exp=transglower_aistpp_expmap
#exp=transglower_residual_aistpp_expmap
#exp=transflower_residual_aistpp_expmap
exp=mowgli_aistpp_expmap
#exp=transflower_aistpp_expmap
#exp=residualflower2_transflower_aistpp_expmap
#exp=moglow_aistpp_expmap
hparams_file=aistpp_20hz/${exp}
exp=mowgli_aistpp_expmap_future3

## Fix: needs vmapped version of transformer:
#hparams_file=aistpp_20hz/residualflower2_moglow_aistpp_expmap

####moglow_pos
#data_dir=${root_dir}/moglow_pos
#exp=$1
#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos
#exp=moglow_trans_moglow_pos
#hparams_file=moglow_pos/${exp}
#exp=testing
#exp=${exp}_pos_emb

#exp=${exp}_future3_actnorm
#exp=${exp}_future3
#exp=testing

echo $exp

$py training/train.py --data_dir=${data_dir} --max_epochs=2000\
    --do_validation \
    --hparams_file=training/hparams/${hparams_file}.yaml \
    --val_batch_size=2 \
    --batch_size=8 \
    --experiment_name=$exp\
    --workers=$(nproc) \
    --gpus=1 \
    --accelerator=ddp \
    --output_lengths="3" \
    --stage2 \
    --continue_train \
    --load_weights_only \
    --prior_use_x_transformers \
#    --scales="[[16,0]]" \
#    --use_x_transformers \
#    --use_rotary_pos_emb \
#    --learning_rate=1e-5 \
    #--residual_scales="[[16,0]]"
#    --glow_norm_layer="actnorm" \
    #--use_pos_emb_output \
#    --tpu_cores=8 \
