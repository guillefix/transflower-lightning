import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse
import time
import json, pickle
import torch
from math import ceil
from scipy import signal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(THIS_DIR)

import models.constants as constants
from models import create_model
from scripts.feature_extraction.utils import distribute_tasks,ResampleLinear1D

parser = argparse.ArgumentParser(description='Get DDC features from song features')
parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument('--checkpoints_dir', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--peak_threshold', type=float, default=0.0148)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--cuda', action="store_true")
parser.add_argument("--step_size", metavar='', type=float, default=0.01666666666)
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
# print("creating {} of size {}".format(feature_name, feature_size))

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature

from pathlib import Path

''' LOAD MODEL, OPTS, AND WEIGHTS'''
#%%

##loading opt object from experiment
opt = json.loads(open(ROOT_DIR.__str__()+"/scripts/training/"+experiment_name+"opt.json","r").read())
# we assume we have 1 GPU in generating machine :P
if args.cuda:
    opt["gpu_ids"] = [0]
else:
    opt["gpu_ids"] = []
opt["checkpoints_dir"] = args.checkpoints_dir
opt["load_iter"] = int(checkpoint)
if args.cuda:
    opt["cuda"] = True
else:
    opt["cuda"] = False
opt["experiment_name"] = args.experiment_name.split("/")[0]
if "dropout" not in opt: #for older experiments
    opt["dropout"] = 0.0
# construct opt Struct object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

assert opt.binarized

model = create_model(opt)
model.setup()
receptive_field = 1

checkpoint = "iter_"+checkpoint
model.load_networks(checkpoint)

#assuming mp3 for now. TODO: generalize
candidate_feature_files = sorted(data_path.glob('**/*mp3_multi_mel_80.npy'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_feature_files,rank,size)

for i in tasks:
    path = candidate_feature_files[i]
    features_file = str(path)+"_"+"ddc_hidden"+".npy"
    print(path)

    sr = opt.sampling_rate
    hop = int(opt.step_size*sr)
    features = np.load(path)

    #generate level
    # first_samples basically works as a padding, for the first few outputs, which don't have any "past part" of the song to look at.
    first_samples = torch.full((1,opt.output_channels,receptive_field//2),constants.START_STATE,dtype=torch.float)
    print(features.shape)
    features, peak_probs = model.generate_features(features)
    peak_probs = peak_probs[0,:,-1].cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    features = features[0]
    features = ResampleLinear1D(features,int(np.floor(features.shape[0]*0.01/0.016666666)))
    # features = downsample_signal(features[0], 0.01666666666667/0.01)
    print(features.shape)
    np.save(features_file,features)
    window = signal.hamming(ceil(constants.HUMAN_DELTA/opt.step_size))
    smoothed_peaks = np.convolve(peak_probs,window,mode='same')

    thresholded_peaks = smoothed_peaks*(smoothed_peaks>args.peak_threshold)
    peaks = signal.find_peaks(thresholded_peaks)[0]
    print("number of peaks", len(peaks))
