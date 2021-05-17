#!/bin/bash

find $1 -name "*.wav" -exec ffprobe -v quiet -of csv=p=0 -show_entries format=duration {} \; | paste -sd+ -| bc | xargs -I {} echo "scale=4; {}/60.0" | bc
