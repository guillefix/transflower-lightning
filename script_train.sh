#!/bin/bash

#export TPU_IP_ADDRESS=10.8.195.90;
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

py=python3

#data_dir=data/moglow_loc

####aistpp_60hz
#data_dir=data/scaled_features
#hparams_file=aistpp_60hz/transflower_aistpp_expmap
#hparams_file=aistpp_60hz/transglower_aistpp_expmap

####aistpp_20hz
#data_dir=data/features_20
#hparams_file=aistpp_20hz/transglower_aistpp_expmap
#hparams_file=aistpp_20hz/transglower_residual_aistpp_expmap
#hparams_file=aistpp_20hz/transflower_residual_aistpp_expmap
#hparams_file=aistpp_20hz/transflower_aistpp_expmap
#hparams_file=aistpp_20hz/residualflower2_transflower_aistpp_expmap
#hparams_file=aistpp_20hz/moglow_aistpp_expmap

## Fix: needs vmapped version of transformer:
#hparams_file=aistpp_20hz/residualflower2_moglow_aistpp_expmap

####moglow_pos
data_dir=data/moglow_loc
hparams_file=moglow_pos/transglower_moglow_pos
#hparams_file=moglow_pos/transglower_residual_moglow_pos
#hparams_file=moglow_pos/transflower_residual_moglow_pos
#hparams_file=moglow_pos/transflower_moglow_pos
#hparams_file=moglow_pos/residualflower2_transflower_moglow_pos
#hparams_file=moglow_pos/moglow_moglow_pos

exp=testing

$py training/train.py --data_dir=${data_dir} --max_epochs=50\
    --do_testing \
    --hparams_file=training/hparams/${hparams_file}.yaml \
    --val_batch_size=2 \
    --batch_size=8 \
    --experiment_name=$exp\
    --workers=$(nproc) \
    --gpus=1 \
#    --accelerator=ddp \
#    --continue_train \
#    --tpu_cores=8 \