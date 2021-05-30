import numpy as np
# import librosa
from pathlib import Path
import json
import os.path
import sys
import argparse
import pickle
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(ROOT_DIR)
from utils import distribute_tasks

from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import json

parser = argparse.ArgumentParser(description="Extract features from filenames")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--files_extension", type=str, help="file extension (the stuff after the base filename) to match")
parser.add_argument("--name_processing_function", type=str, default="dance_style", help="function for processing the names")
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

assert size == 1 # this should be done with one process

files = sorted(data_path.glob('**/*.'+files_extension), key=lambda path: path.parent.__str__())
# tasks = distribute_tasks(candidate_motion_files,rank,size)

import name_processing_functions
func = getattr(name_processing_functions, name_processing_function)
labels = list(map(func,files))
unique_labels = np.unique(list(labels))
print(unique_labels)
label_index = {c:i for i,c in enumerate(unique_labels)}
label_index_reverse = {i:c for i,c in enumerate(unique_labels)}
with open(str(data_path) + "/" + files_extension+"."+name_processing_function+'class_index.json', 'w') as f:
    json.dump(label_index, f)
with open(str(data_path) + "/" + files_extension+"."+name_processing_function+'class_index_reverse.json', 'w') as f:
    json.dump(label_index_reverse, f)

for file,label in zip(files,labels):
    # print(file, label)
    feature_name = str(file)+"."+name_processing_function
    feature = np.array([label_index[label]])
    np.save(feature_name, feature)
