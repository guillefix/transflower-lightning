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
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)
from utils import distribute_tasks

parser = argparse.ArgumentParser(description="Preprocess motion data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--replace_existing", action="store_true")

args = parser.parse_args()

# makes arugments into global variables of the same name, used later in the code
globals().update(vars(args))
data_path = Path(data_path)

from scipy.spatial.transform import Rotation as R
from typing import NewType, Union, Optional
Tensor = NewType('Tensor', torch.Tensor)
def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def get_rot_matrices_from_euler(joint_traj):
    return np.stack([np.concatenate([R.from_euler('xyz',euler_angles).as_matrix().flatten() for euler_angles in np.array(joint_angles).reshape(-1,3)]) for joint_angles in joint_traj])

def get_rot_matrices_from_axis_angle(joint_traj):
    # return np.stack([np.concatenate([R.from_euler('xyz',euler_angles).as_matrix().flatten() for euler_angles in np.array(joint_angles).reshape(-1,3)]) for joint_angles in joint_traj])
    L = len(joint_traj)
    ident = torch.eye(3, dtype=torch.float32)
    rot_mats = batch_rodrigues(torch.from_numpy(joint_traj).view(-1, 3)).view(
        [L,-1, 3, 3])
    pose_feature = (rot_mats - ident).view([L, -1])
    return pose_feature.numpy()

def get_features(motion_data):
    joint_angle_feats = get_rot_matrices_from_axis_angle((motion_data['smpl_poses']))
    return np.concatenate([joint_angle_feats,motion_data['smpl_trans']],1)

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
    features_file = motion_file_path+"_"+"joint_angles_mats"+".npy"
    if replace_existing or not os.path.isfile(features_file):
        motion_data = pickle.load(open(path,"rb"))
        features = get_features(motion_data)
        print(features.shape)
        features = ResampleLinear1D(features,features.shape[0]*2)
        print(features.shape)
        np.save(features_file,features)
