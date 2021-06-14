#!/bin/bash

root_dir=dance_combined3
target_dir=dance_combined3
#gsutil -m cp ${root_dir}/*expmap_scaled_20* gs://metagen/data/${target_dir}
gsutil -m cp ${root_dir}/*expmap_cr_scaled_20* gs://metagen/data/${target_dir}
gsutil -m cp ${root_dir}/*audio_feats_scaled_20* gs://metagen/data/${target_dir}
gsutil -m cp ${root_dir}/*.pkl gs://metagen/data/${target_dir}
gsutil -m cp ${root_dir}/*.sav gs://metagen/data/${target_dir}
gsutil -m cp ${root_dir}/*.json gs://metagen/data/${target_dir}
gsutil -m cp ${root_dir}/base_filenames* gs://metagen/data/${target_dir}
