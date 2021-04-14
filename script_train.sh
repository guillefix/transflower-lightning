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

root_dir=$SCRATCH/data

####aistpp_60hz
#data_dir=${root_dir}/scaled_features
#hparams_file=aistpp_60hz/transflower_aistpp_expmap
#hparams_file=aistpp_60hz/transglower_aistpp_expmap

####aistpp_20hz
#data_dir=${root_dir}/features_20
#hparams_file=aistpp_20hz/transglower_aistpp_expmap
#hparams_file=aistpp_20hz/transglower_residual_aistpp_expmap
#hparams_file=aistpp_20hz/transflower_residual_aistpp_expmap
#hparams_file=aistpp_20hz/transflower_aistpp_expmap
#hparams_file=aistpp_20hz/residualflower2_transflower_aistpp_expmap
#hparams_file=aistpp_20hz/moglow_aistpp_expmap

## Fix: needs vmapped version of transformer:
#hparams_file=aistpp_20hz/residualflower2_moglow_aistpp_expmap

####moglow_pos
data_dir=${root_dir}/moglow_pos
exp=$1
#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos

hparams_file=moglow_pos/${exp}


$py training/train.py --data_dir=${data_dir} --max_epochs=2000\
    --do_testing \
    --hparams_file=training/hparams/${hparams_file}.yaml \
    --val_batch_size=8 \
    --batch_size=32 \
    --experiment_name=$exp\
    --workers=$(nproc) \
    --gpus=2 \
    --accelerator=ddp \
#    --continue_train \
#    --tpu_cores=8 \
