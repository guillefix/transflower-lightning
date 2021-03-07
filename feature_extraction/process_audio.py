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
from audio_feature_utils import extract_features_hybrid, extract_features_mel, extract_features_multi_mel, extract_features_envelope
from utils import distribute_tasks

parser = argparse.ArgumentParser(description="Preprocess audio data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_name", metavar='', type=str, default="mel", help="mel, chroma, multi_mel")
parser.add_argument("--feature_size", metavar='', type=int, default=100)
parser.add_argument("--step_size", metavar='', type=float, default=0.01666666666)
parser.add_argument("--sampling_rate", metavar='', type=float, default=96000)
parser.add_argument("--replace_existing", action="store_true")
parser.add_argument("--notranspose", action="store_true")

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
print("creating {} of size {}".format(feature_name, feature_size))

#assuming mp3 for now.
candidate_audio_files = sorted(data_path.glob('**/*.mp3'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_audio_files,rank,size)

for i in tasks:
    path = candidate_audio_files[i]
    song_file_path = path.__str__()
    # feature files are going to be saved as numpy files
    features_file = song_file_path+"_"+feature_name+"_"+str(feature_size)+".npy"

    if replace_existing or not os.path.isfile(features_file):
        print("creating feature file",i)

        # get song
        y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)

        sr = sampling_rate
        hop = int(round(sr * step_size))

        #get feature
        if feature_name == "chroma":
            features = extract_features_hybrid(y_wav,sr,hop)
        elif feature_name == "mel":
            features = extract_features_mel(y_wav,sr,hop,mel_dim=feature_size)
        elif feature_name == "envelope":
            features = extract_features_envelope(y_wav,sr,hop,mel_dim=feature_size)
        elif feature_name == "multi_mel":
            features = extract_features_multi_mel(y_wav, sr=sampling_rate, hop=hop, nffts=[1024,2048,4096], mel_dim=feature_size)

        if notranspose:
            np.save(features_file,features)
        else:
            np.save(features_file,features.transpose(1,0))
