#!/bin/bash

augmentation=mirrored
find $1 -name "*.${2}.npy" -not -name "*_${augmentation}*.${2}.npy" -print0 | xargs -0 -I {} basename -z {} .${2}.npy | xargs -0 -I {} cp $1/{}.${2}.npy $1/{}_${augmentation}.${2}.npy
#find $1 -name "*.${2}.npy" -print0 | xargs -0 -I {} basename -z {} .${2}.npy | xargs -0 -I {} echo {}_${augmentation}.${2}.npy
#find $1 -name "*.${2}.npy" -print0 | xargs -0 -I {} basename {} .${2}.npy
