#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6

$py ./feature_extraction/process_filenames.py $1 --files_extension bvh --name_processing_function dance_style ${@:2}
