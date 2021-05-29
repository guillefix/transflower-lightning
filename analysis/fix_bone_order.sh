#!/bin/bash

# I think call this first before fix_scales.sh

# find $1 -name "*.bvh" -print0 | xargs -0 -I{} python3 analysis/fix_bone_order.py {}
find $1 -name "*.bvh" -print0 | parallel -0 -I{} python3 analysis/fix_bone_order.py {}
