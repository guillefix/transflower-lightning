#!/bin/bash
exp=$1
mkdir training/experiments/${exp}
scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/${exp}/* training/experiments/${exp}
