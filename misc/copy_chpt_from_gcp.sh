#!/bin/bash
instance=$1
exp=$2
version=$3
gcloud=gcloud
#gcloud=$SCRATCH/google-cloud-sdk/bin/gcloud
mkdir training/experiments/${exp}
mkdir training/experiments/${exp}/version_${version}
#scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/inference/generated/${exp}/videos/* inference/generated/${exp}/videos
$gcloud beta compute scp --recurse --zone "europe-west4-a" ${instance}:~/mt-lightning/training/experiments/${exp}/version_${version}/* --project "kumofix2" training/experiments/${exp}/version_${version}
