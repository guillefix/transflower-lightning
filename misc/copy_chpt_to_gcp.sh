#!/bin/bash
instance=$1
exp=$2
gcloud=gcloud
#gcloud=$SCRATCH/google-cloud-sdk/bin/gcloud
#mkdir training/experiments/${exp}
#scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/inference/generated/${exp}/videos/* inference/generated/${exp}/videos
$gcloud beta compute scp --recurse --zone "europe-west4-a" ./training/experiments/${exp} ${instance}:~/mt-lightning/training/experiments/${exp}/ --project "kumofix2" 
