#!/bin/bash
file=$1
format=mp4
times_file=$2
timestamp() {
    date '+%s' --date="$1";
}
time1=00:00:00
i=0
while read -r line;
do
    echo $i
    echo $line
    time2=$line
    t=$(($(timestamp $time2)-$(timestamp $time1)))
    echo $t
    ffmpeg -hide_banner -loglevel error -nostdin -y -i $file -ss $time1 -t $t $(basename $file .$format)_${i}.$format
    time1=$time2
    i=$(($i+1))
done < $times_file
