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
import joblib as jl

parser = argparse.ArgumentParser(description="Preprocess motion data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--param", type=str, default="expmap", help="expmap, position")
parser.add_argument("--replace_existing", action="store_true")
parser.add_argument("--do_mirror", action="store_true", help="whether to augment the data with mirrored motion")
parser.add_argument("--fps", type=int, default=60)

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

p = BVHParser()
if do_mirror:
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
        ('mir', Mirror(axis='X', append=True)),
        ('root', RootTransformer('pos_rot_deltas')),
        # ('jtsel', JointSelector(['Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
        ('jtsel', JointSelector(['Spine', 'Spine1', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
        (param, MocapParameterizer(param)),
        ('cnst', ConstantsRemover(only_cols=["Hips_Xposition", "Hips_Zposition"])),
        ('np', Numpyfier())
    ])
else:
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
        ('root', RootTransformer('pos_rot_deltas')),
        # ('mir', Mirror(axis='X', append=True)),
        # ('jtsel', JointSelector(['Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
        ('jtsel', JointSelector(['Spine', 'Spine1', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
        (param, MocapParameterizer(param)),
        ('cnst', ConstantsRemover(only_cols=["Hips_Xposition", "Hips_Zposition"])),
        ('np', Numpyfier())
    ])

def extract_joint_angles(files):
    if len(files)>0:
        data_all = list()
        for f in files:
            data_all.append(p.parse(f))

        out_data = data_pipe.fit_transform(data_all)

        if do_mirror:
            # NOTE: the datapipe will append the mirrored files to the end
            assert len(out_data) == 2*len(files)
        else:
            assert len(out_data) == len(files)

        if rank == 0:
            jl.dump(data_pipe, os.path.join(data_path, 'motion_'+param+'_data_pipe.sav'))

        fi=0
        if do_mirror:
            for f in files:
                features_file = f + "_"+param+".npy"
                if replace_existing or not os.path.isfile(features_file):
                    np.save(features_file, out_data[fi])
                features_file_mirror = f[:-4]+"_mirrored" + ".bvh_"+param+".npy"
                if replace_existing or not os.path.isfile(features_file_mirror):
                    np.save(features_file_mirror, out_data[len(files)+fi])
                fi=fi+1
        else:
            for f in files:
                features_file = f + "_"+param+".npy"
                if replace_existing or not os.path.isfile(features_file):
                    np.save(features_file, out_data[fi])
                fi=fi+1

candidate_motion_files = sorted(data_path.glob('**/*.bvh'), key=lambda path: path.parent.__str__())
#candidate_motion_files = candidate_motion_files[:32]
tasks = distribute_tasks(candidate_motion_files,rank,size)

files = [path.__str__() for i, path in enumerate(candidate_motion_files) if i in tasks]

extract_joint_angles(files)
