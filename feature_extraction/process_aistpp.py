import numpy as np
# import librosa
from pathlib import Path
import json
import os.path
import sys
import argparse
import pickle
import torch
import ntpath

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
ANALYSIS_DIR = os.path.join(ROOT_DIR, 'analysis')
if not os.path.isdir(ANALYSIS_DIR):
    os.mkdir(ANALYSIS_DIR)
sys.path.append(ROOT_DIR)
from utils import distribute_tasks
from utils.smpl2mixamo import convert_smpl

parser = argparse.ArgumentParser(description="Preprocess motion data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
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

candidate_motion_files = sorted(data_path.glob('**/*.pkl'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_motion_files,rank,size)

for i in tasks:
    path = candidate_motion_files[i]
    motion_file_path = path.__str__()
    bvh_file = motion_file_path[:-4]+".bvh"
    if replace_existing or not os.path.isfile(bvh_file):
        result_filename=ntpath.basename(motion_file_path)[:-4]
        print("retargetting "+motion_file_path)
        convert_smpl(motion_file_path, data_path, result_filename,60)

