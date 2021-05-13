import librosa
import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse

'''
Compute transforms which can be computed sequentially (so they implement the `partial_fit` function)
'''

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)

parser = argparse.ArgumentParser(description="Preprocess songs data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_name", metavar='', type=str, default="mel", help="mel, chroma, multi_mel")
parser.add_argument("--transforms", metavar='', type=str, default="scaler", help="comma-separated lists of transforms to extract (scaler,pca_transform)")
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
assert size == 1
candidate_files = sorted(data_path.glob('**/*'+feature_name+'.npy'), key=lambda path: path.parent.__str__())
tasks = range(len(candidate_files))

from sklearn import decomposition, preprocessing
import pickle
transforms = transforms.split(",")
transforms_dict = {}
for transform in transforms:
    if transform == "scaler":
        scaler = preprocessing.StandardScaler()
        transforms_dict["scaler"] = scaler
    elif transform == "pca_transform":
        features = np.load(candidate_files[0].__str__())
        feature_size = features.shape[1]
        pca = decomposition.PCA(n_components=feature_size)
        transforms_dict["pca_transform"] = pca
    else:
        raise NotImplementedError("Transform type "+transform+" not implemented")
for i in tasks:
    path = candidate_files[i]
    feature_file = path.__str__()
    features = np.load(feature_file)
    for transform in transforms:
        if len(features.shape) == 3:
            features = features[:,0,:]
        transforms_dict[transform].partial_fit(features)

for transform in transforms:
    pickle.dump(transforms_dict[transform], open(data_path.joinpath(feature_name+'_'+transform+'.pkl'), 'wb'))
