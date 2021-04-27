#!/bin/bash
instance=$1
exp=$2
mkdir inference/generated
mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/videos
#scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/inference/generated/${exp}/videos/* inference/generated/${exp}/videos
gcloud beta compute scp --recurse --zone "europe-west4-a" ${instance}:~/mt-lightning/inference/generated/${exp}/videos/* --project "kumofix2" inference/generated/${exp}/videos/
