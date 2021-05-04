#!/bin/bash

#data_dir=.
data_dir=$SCRATCH/data
#gsutil=gsutil
gsutil=$SCRATCH/google-cloud-sdk/bin/gsutil

#gsutil -m cp -r gs://metagen/data/moglow_pos $data_dir/
#gsutil -m cp -r gs://metagen/data/aistpp_20hz $data_dir/
#$gsutil -m cp -r gs://metagen/data/dance_combined $data_dir/
#gsutil -m cp -r gs://metagen/data/aistpp_60hz $data_dir/
#$gsutil -m cp -r gs://metagen/data/aistpp/music/*.mp3 $data_dir/aistpp_20hz/
#$gsutil -m -o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24 cp -r gs://metagen/data/aistpp/music/*.mp3 $data_dir/aistpp_20hz/
#$gsutil -m -o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24 cp -r gs://metagen/data/dance_combined $data_dir/
#$gsutil -m -o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24 cp -r gs://metagen/data/dance_combined/justdance* $data_dir/dance_combined
#$gsutil -m -o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24 cp -r gs://metagen/data/dance_combined/base_filenames* $data_dir/dance_combined
$gsutil -m -o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24 cp -r gs://metagen/data/dance_combined/kthmisc_10.audio_feats_scaled_20.npy $data_dir/dance_combined

#gsutil -m cp gs://metagen/data/features_20/*mel_ddcpca_scaled_20* features/
#gsutil -m cp gs://metagen/data/features_20/*scaler* features/
#gsutil -m cp gs://metagen/data/features_20/*data_pipe* features/
#gsutil -m cp gs://metagen/data/features_20/*base_filenames* features/

#if you don't have gsutil then use this for aistpp_20hz
#curl -L https://kth.box.com/shared/static/zd4b27jhrn819vkzlvmpkuhofaehslvo.gz --output features_20.tar.gz

# but you can download gsutil here curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-336.0.0-linux-x86_64.tar.gz
# and follow instructions here https://cloud.google.com/storage/docs/gsutil_install#linux 
