#!/bin/bash
exp=$1
scp -r training/experiments/${exp}/* jeanzay:/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/${exp}/
