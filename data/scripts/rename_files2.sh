#!/bin/bash

mkdir ${4}
OIFS="$IFS"
IFS=$'\n'

j=38
#j=0
i=1
for file in `find ${1} -type f -name "*.${3}" | sort -t'_' -k 2 -n`; do
    #bn=$(basename "$f" .bvh)
    cp $file ${4}/${2}$(($i+$j))_ch$(printf "%02d" $(($i+$j))).${3}
    #cp $file renamed/${2}_${i}.${3}
    echo $(($i+$j-1)),$file
    #echo $i
    i=$(($i+1))
done
#for i in `seq $2 -1 $3`; do
#    cp ${1}/${4}_${i}.${5} renamed/${6}_$(($i+$j))_ch$(printf "%02d" $(($i+$j))).${5}
#    echo ${4}_${i}.${5} -> ${6}_$(($i+$j))_ch$(printf "%02d" $(($i+$j))).${5}
#done


