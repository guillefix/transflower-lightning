#!/bin/bash

mkdir renamed
OIFS="$IFS"
IFS=$'\n'

i=0
for file in `find ${1} -type f -name "*.${3}" | sort`; do
    #bn=$(basename "$f" .bvh)
    cp $file renamed/${2}_${i}.${3}
    echo $i,$file
    #echo $i
    i=$(($i+1))
done


