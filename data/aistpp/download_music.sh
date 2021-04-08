#!/bin/bash

DOWNLOAD_FOLDER=videos
mkdir $DOWNLOAD_FOLDER
wget https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py
python3 downloader.py --download_folder=$DOWNLOAD_FOLDER --num_processes=$(nproc)

find $DOWNLOAD_FOLDER -type f -name '*.mp4' -print0 | parallel -0 ffmpeg -i {} {.}.wav

mkdir music
cp $DOWNLOAD_FOLDER/*c01*.wav music
rename 's/c01/cAll/g' music/*c01*.wav

