#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6

$py ./feature_extraction/process_filenames.py $1 --files_extension expmap_cr_scaled_20.npy --name_processing_function dance_style ${@:2}
find $1 -exec rename 's/expmap_cr_scaled_20.npy.dance_style/dance_style/' {} +
