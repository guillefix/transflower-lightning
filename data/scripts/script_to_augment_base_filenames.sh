#!/bin/bash

file=$1
augmentation=mirrored
cat $file | xargs -I {} echo {}_${augmentation} > ${file}_mirrored
cat $file ${file}_mirrored > ${file}_new
mv ${file}_new $file
