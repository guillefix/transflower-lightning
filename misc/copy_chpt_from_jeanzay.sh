#!/bin/bash
exp=$1
version=$2
mkdir training/experiments/${exp}
mkdir training/experiments/${exp}/version_${version}
scp -r jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/${exp}/version_${version}/* training/experiments/${exp}/version_${version}
