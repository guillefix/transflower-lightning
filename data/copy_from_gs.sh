#!/bin/bash

gsutil -m cp gs://metagen/data/features_20/*expmap_scaled_20* features/
gsutil -m cp gs://metagen/data/features_20/*mel_ddcpca_scaled_20* features/
gsutil -m cp gs://metagen/data/features_20/*scaler* features/
gsutil -m cp gs://metagen/data/features_20/*data_pipe* features/
gsutil -m cp gs://metagen/data/features_20/*base_filenames* features/

