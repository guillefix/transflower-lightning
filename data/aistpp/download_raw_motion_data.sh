#!/bin/bash

#wget http://storage.cloud.google.com/aist_plusplus_public/20210308/fullset.zip
gsutil cp gs://aist_plusplus_public/20210308/fullset.zip .
unzip fullset.zip
