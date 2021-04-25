import librosa
import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
from audio_feature_utils import *
from utils import distribute_tasks

parser = argparse.ArgumentParser(description="Preprocess audio data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_names", metavar='', type=str, default="mel", help="mel, chroma, multi_mel, spectralflux, madmombeats. Comma separated")
parser.add_argument("--combined_feature_name", metavar='', type=str, default=None, help="name for the combined features, if several")
parser.add_argument("--audio_format", type=str, default="mp3")
parser.add_argument("--mel_feature_size", metavar='', type=int, default=None)
# parser.add_argument("--step_size", metavar='', type=float, default=0.01666666666)
parser.add_argument("--fps", metavar='', type=float, default=60)
parser.add_argument("--sampling_rate", metavar='', type=float, default=96000)
parser.add_argument("--replace_existing", action="store_true")
parser.add_argument("--notranspose", action="store_true")

args = parser.parse_args()

# makes arugments into global variables of the same name, used later in the code
globals().update(vars(args))
step_size=1.0/fps
data_path = Path(data_path)

feature_names = feature_names.split(",")
if len(feature_names) > 1 and combined_feature_name is None:
    combined_feature_name = "_".join(feature_names)
elif len(feature_names) == 1:
    combined_feature_name = feature_names[0]

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
print("creating {} of size {}".format(",".join(feature_names), mel_feature_size))

#assuming mp3 for now.
candidate_audio_files = sorted(data_path.glob('**/*.'+audio_format), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_audio_files,rank,size)

for i in tasks:
    path = candidate_audio_files[i]
    song_file_path = path.__str__()
    # feature files are going to be saved as numpy files
    if feature_names == ["mel"] or feature_names == ["multi_mel"]:
        features_file = song_file_path+"_"+combined_feature_name+"_"+str(mel_feature_size)+".npy"
    else:
        features_file = song_file_path+"_"+combined_feature_name+".npy"

    if replace_existing or not os.path.isfile(features_file):
        print("creating feature file",i)
        featuress = []
        for feature_name in feature_names:
            # get song
            y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)

            sr = sampling_rate
            hop = int(round(sr * step_size))
            # hop = int(sr * step_size)

            #get feature
            if feature_name == "chroma":
                features = extract_features_hybrid(y_wav,sr,hop).transpose(1,0)[1:]
            elif feature_name == "mel":
                features = extract_features_mel(y_wav,sr,hop,mel_dim=mel_feature_size).transpose(1,0)[1:]
            elif feature_name == "envelope":
                features = extract_features_envelope(y_wav,sr,hop)[1:]
            elif feature_name == "multi_mel":
                features = extract_features_multi_mel(y_wav, sr=sampling_rate, hop=hop, nffts=[1024,2048,4096], mel_dim=mel_feature_size)
            elif feature_name == "spectralflux": #actually this is the same as envelope I think
                features = extract_features_spectral_flux(song_file_path,fps)
            elif feature_name == "madmombeats":
                features = extract_features_madmombeat(song_file_path,fps)

            featuress.append(features)

        shortest_length = 99999999999
        for feat in featuress:
            if feat.shape[0] < shortest_length:
                shortest_length = feat.shape[0]
        for i in range(len(featuress)):
            featuress[i] = featuress[i][:shortest_length]

        featuress = np.concatenate(featuress,1)

        np.save(features_file,featuress)
