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
exp=$1

####aistpp_60hz
#data_dir=${root_dir}/scaled_features
#hparams_file=aistpp_60hz/${exp}

####aistpp_20hz
#data_dir=${root_dir}/aistpp_20hz
#hparams_file=aistpp_20hz/${exp}

####moglow_pos
#data_dir=${root_dir}/moglow_pos
#hparams_file=moglow_pos/${exp}

####dance_combined
#data_dir=${root_dir}/dance_combined
#data_dir=${root_dir}/dance_combined2
data_dir=${root_dir}/dance_combined3
hparams_file=dance_combined/${exp}

echo $exp
#echo $RANK
#echo $LOCAL_RANK
echo $SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

$py training/train.py --data_dir=${data_dir} \
    --max_epochs=1000\
    --hparams_file=training/hparams/${hparams_file}.yaml \
    --experiment_name=$exp\
    --workers=$(nproc) \
    --gpus=-1 \
    --accelerator=ddp \
    ${@:2} #NOTE: can override experiment_name, and any of the options above
    #--batch_size=32 \
    #--plugins=deepspeed \
    #--precision=16 \

    #--gradient_clip_val=0.5 \
    #--sync_batchnorm \
    #--lr_policy=LinearWarmupCosineAnnealing \
    #--auto_lr_find \
    #--do_tuning \
    #--learning_rate=7e-5 \
    #--batch_size=84 \
    #--num_nodes=4 \
    #--output_lengths=3 \
    #--dropout=0.1 \
    #--vae_dhid=128 \
    #--optimizer=madgrad \
    #--learning_rate=1e-3 \
    #--use_x_transformers \
    #--use_rotary_pos_emb \
    #--batch_size=84 \
    #--lr_policy=reduceOnPlateau \

    #--learning_rate=1e-4 \
    #--use_pos_emb_output \
    #--flow_dist=studentT \
    #--gradient_clip_val=1 \
    #--flow_dist=studentT \
    #--fix_lengths \
    #--use_x_transformers \
    #--use_rotary_pos_emb \
    #--output_lengths="3" \
    #--scales="[[16,0]]" \
    #--residual_scales="[[16,0]]"
#    --glow_norm_layer="actnorm" \
    #--use_pos_emb_output \
#    --tpu_cores=8 \
