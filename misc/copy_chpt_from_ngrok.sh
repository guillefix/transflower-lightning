#!/bin/bash
n=$1
port=$2
exp=$3
#gcloud=gcloud
gcloud=$SCRATCH/google-cloud-sdk/bin/gcloud
mkdir training/experiments/${exp}
#scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/inference/generated/${exp}/videos/* inference/generated/${exp}/videos
scp -P $port $n.tcp.ngrok.io:~/mt-lightning/training/experiments/${exp}/* training/experiments/
