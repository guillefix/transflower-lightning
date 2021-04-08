#!/bin/bash

mkdir ../features
gsutil -m cp gs://metagen/data/scaled_features/* ../features
