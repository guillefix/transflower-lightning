import librosa
import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)
from audio_feature_utils import extract_features_hybrid, extract_features_mel, extract_features_multi_mel
from utils import distribute_tasks

parser = argparse.ArgumentParser(description="Preprocess songs data")

parser.add_argument("data_path", type=str, help="features path")
parser.add_argument("base_filenames_file", type=str, help="File listing the base names for the files for which to combine features")
parser.add_argument("--feature_names", metavar='', type=str, default="mel", help="coma separated list of names of features to combine")
parser.add_argument("--new_feature_name", metavar='', type=str, default="combined", help="new name for combined feature")
parser.add_argument("--replace_existing", action="store_true")
args = parser.parse_args()

# makes arugments into global variables of the same name, used later in the code
globals().update(vars(args))
data_path = Path(data_path)

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

candidate_files = [x[:-1] for x in open(base_filenames_file,"r").readlines()]
tasks = distribute_tasks(candidate_files,rank,size)

for i in tasks:
    path = candidate_files[i]
    base_filename = data_path.joinpath(path).__str__()
    new_feature_file = base_filename+"."+new_feature_name+".npy"
    if replace_existing or not os.path.isfile(new_feature_file):
        features = None
        for i,feature_name in enumerate(feature_names.split(",")):
            feature_file = base_filename+"."+feature_name+".npy"
            if i == 0:
                features = np.load(feature_file)
            else:
                feature = np.load(feature_file)
                if len(features) > len(feature):
                    features = features[:-1]
                if len(feature) > len(features):
                    feature = feature[:-1]
                if len(feature.shape) == 2:
                    features = np.concatenate([features,feature],1)
                elif len(feature.shape) == 1:
                    features = np.concatenate([features,np.expand_dims(feature,1)],1)
                else:
                    raise NotImplementedError("Only supporting features of rank 1")
        np.save(new_feature_file,features)
