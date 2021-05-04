#!/bin/bash

augmentation=mirrored
find $1 -name "*.${2}.npy" -print0 | xargs -0 -I {} basename {} .${2}.npy | xargs -0 -I {} cp {}.${2}.npy {}_${augmentation}.${2}.npy
