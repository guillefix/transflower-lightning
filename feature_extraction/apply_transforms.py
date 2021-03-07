import librosa
import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse
import pickle

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
parser.add_argument("--feature_name", metavar='', type=str, default="mel", help="coma separated list of names of features to combine")
parser.add_argument("--transform_name", metavar='', type=str, default="scaler", help="pca_transform,scaler")
parser.add_argument("--pca_dims", metavar='', type=int, default=2, help="number of pca dimensions to keep, if applying pca transform")
parser.add_argument("--keep_feature_name", action="store_true")
parser.add_argument("--new_feature_name", metavar='', type=str, default=None)
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

#assuming mp3 for now. TODO: generalize
candidate_files = sorted(data_path.glob('**/*'+feature_name+'.npy'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_files,rank,size)

for i in tasks:
    path = candidate_files[i]
    feature_file = path.__str__()
    if new_feature_name is None:
        if keep_feature_name:
            new_feature_name = feature_name
        else:
            new_feature_name = feature_name+"_applied_"+transform_name
    base_filename = feature_file[:-(len(feature_name)+4)]
    new_feature_file = base_filename+new_feature_name+".npy"
    if replace_existing or not os.path.isfile(new_feature_file):
        features = np.load(feature_file)
        transform = pickle.load(open(data_path.joinpath(feature_name+'_'+transform_name+'.pkl'), "rb"))
        features = transform.transform(features)
        if transform_name == "pca_transform":
            features = features[:,:pca_dims]
        np.save(new_feature_file,features)
