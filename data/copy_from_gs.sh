#!/bin/bash

mkdir features
gsutil -m cp gs://metagen/data/features_20/*expmap_scaled_20* features/
gsutil -m cp gs://metagen/data/features_20/*mel_ddcpca_scaled_20* features/
gsutil -m cp gs://metagen/data/features_20/*scaler* features/
gsutil -m cp gs://metagen/data/features_20/*data_pipe* features/
gsutil -m cp gs://metagen/data/features_20/*base_filenames* features/

#if you don't have gsutil then use this
#curl -L https://kth.box.com/shared/static/zd4b27jhrn819vkzlvmpkuhofaehslvo.gz --output features_20.tar.gz

