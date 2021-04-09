#!/bin/bash

gsutil -m cp data/features/*expmap_scaled_20* gs://metagen/data/features_20
gsutil -m cp data/features/*mel_ddcpca_scaled_20* gs://metagen/data/features_20
gsutil -m cp data/features/*scaler* gs://metagen/data/features_20
gsutil -m cp data/features/*data_pipe* gs://metagen/data/features_20
gsutil -m cp data/features/*base_filenames* gs://metagen/data/features_20
