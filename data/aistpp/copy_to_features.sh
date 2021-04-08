#!/bin/bash

FEATURES_FOLDER=../features
mkdir $FEATURES_FOLDER

cp aist_plusplus_final/motions/*.pkl $FEATURES_FOLDER
cp music/*.wav $FEATURES_FOLDER
./script_to_list_filenames
cp base_filenames.txt $FEATURES_FOLDER
cp ../../analysis/aistpp_base_filenames_train_filtered.txt $FEATURES_FOLDER/base_filenames_train.txt
