#!/bin/bash
exp=$1
mkdir inference/generated
mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/videos
scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/inference/generated/${exp}/videos/* inference/generated/${exp}/videos
