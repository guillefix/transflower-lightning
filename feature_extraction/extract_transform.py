import librosa
import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse

'''
Compute transforms which r4quire to be fitted on all the data at once rather than sequential (so they dont implement the `partial_fit` function)
'''

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
from audio_feature_utils import extract_features_hybrid, extract_features_mel, extract_features_multi_mel
from utils import distribute_tasks
#from scripts.feature_extraction.utils import distribute_tasks

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
features = None
for i in tasks:
    path = candidate_files[i]
    feature_file = path.__str__()
    if i == 0:
        features = np.load(feature_file)
    else:
        feature = np.load(feature_file)
        features = np.concatenate([features,feature],0)

import pickle
transforms = transforms.split(",")
for transform in transforms:
    if transform == "2moments":
        if len(features.shape) == 3:
            features = features[:,0,:]
        C = np.dot(features.T,features)/features.shape[0]
        m = np.mean(features,0)
        pickle.dump((m,C), open(data_path.joinpath(feature_name+'_2moments.pkl'), 'wb'))
    elif transform == "2moments_ext":
        if len(features.shape) == 3:
            features = features[:,0,:]
        if features.shape[0] % 3 != 0:
            features = features[:-(features.shape[0]%3)]
        features = np.reshape(features,(-1,3*features.shape[1]))
        C = np.dot(features.T,features)/features.shape[0]
        m = np.mean(features,0)
        pickle.dump((m,C), open(data_path.joinpath(feature_name+'_2moments_ext.pkl'), 'wb'))
    elif transform == "scaler":
        scaler = preprocessing.StandardScaler().fit(features)
        pickle.dump(scaler, open(data_path.joinpath(feature_name+'_scaler.pkl'), 'wb'))
    elif transform == "pca_transform":
        feature_size = features.shape[1]
        pca = decomposition.PCA(n_components=feature_size)
        pca_transform = pca.fit(features)
        pickle.dump(pca_transform, open(data_path.joinpath(feature_name+'_pca_transform.pkl'), 'wb'))
    else:
        raise NotImplementedError("Transform type "+transform+" not implemented")
