#!/bin/bash

find videos/ -type f -name '*.mp4' -print0 | parallel -0 ffmpeg -i {} {.}.wav
