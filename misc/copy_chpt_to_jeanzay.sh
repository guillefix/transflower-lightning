#!/bin/bash
exp=$1
version=$2
scp -r training/experiments/${exp}/version_${version}/* jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/${exp}/version_${version}
