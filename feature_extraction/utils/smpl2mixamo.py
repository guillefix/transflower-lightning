
import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('PyMO'))
if sys.path[0] != module_path:
    sys.path.insert(0,module_path)
from analysis.pymo.parsers import BVHParser
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *

import pickle

#%% Default skeleton

p = BVHParser()
default_skel = p.parse("analysis/mixamo.bvh")
#%%

# scale=1.0
# # Scale the default sceleton
# for joint in default_skel.traverse():
#     offs = np.array(default_skel.skeleton[joint]['offsets'])*scale
#     default_skel.skeleton[joint]['offsets'] = offs.tolist()
bvh2expmap = MocapParameterizer('expmap')
default_exp_map_data = bvh2expmap.fit_transform([default_skel])
unity_joints_vals = default_exp_map_data[0].values.columns

npfier = Numpyfier()
npfier.fit_transform(default_exp_map_data)

def angles2smplbvh(out_data, dest_dir, filename, framerate):
    smpl_bvh = bvh2expmap.inverse_transform(npfier.inverse_transform(out_data))
    smpl_bvh[0].framerate=float(1.0/framerate)

    writer = BVHWriter()

    with open(f'{dest_dir}/{filename}.bvh','w') as f:
        writer.write(smpl_bvh[0], f)


def convert_smpl(filepath, dest_dir, result_filename, framerate):
    data = pickle.load(open(filepath, "rb"))

    smpl_angs = data["smpl_poses"].reshape(-1,24*3)
    smpl_trans = data["smpl_trans"]

    smpl_joints = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
                   "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
                   "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"]

    unity_joints = [j for j in default_skel.traverse() if j[-3:]!="Nub"]

    smpl_joints_to_unity_joints = {
        "Pelvis": "Hips",
        "L_Hip": "LeftUpLeg",
        "R_Hip": "RightUpLeg",
        "Spine1": "Spine",
        "L_Knee": "LeftLeg",
        "R_Knee": "RightLeg",
        "L_Ankle": "LeftFoot",
        "R_Ankle": "RightFoot",
        "Spine2": "Spine1",
        "Spine3": "Spine2",
        "L_Foot": "LeftToeBase",
        "R_Foot": "RightToeBase",
        "Neck": "Neck",
        "Head": "Head",
        "L_Collar": "LeftShoulder",
        "R_Collar": "RightShoulder",
        "L_Shoulder": "LeftArm",
        "R_Shoulder": "RightArm",
        "L_Elbow": "LeftForeArm",
        "R_Elbow": "RightForeArm",
        "L_Wrist": "LeftHand",
        "R_Wrist": "RightHand",
        # "L_Hand": "LeftHand",
        # "R_Hand": "RightHand",
    }

    out_data = np.zeros((1, smpl_angs.shape[0], len(unity_joints)*3 + 3))
    poss = np.zeros((smpl_angs.shape[0], 3))
    for smpl_joint, unity_joint in smpl_joints_to_unity_joints.items():
        out_data[0,:,unity_joints_vals.get_loc(unity_joint+"_alpha")] = smpl_angs[:,smpl_joints.index(smpl_joint)*3]
        out_data[0,:,unity_joints_vals.get_loc(unity_joint+"_beta")] = smpl_angs[:,smpl_joints.index(smpl_joint)*3+1]
        out_data[0,:,unity_joints_vals.get_loc(unity_joint+"_gamma")] = smpl_angs[:,smpl_joints.index(smpl_joint)*3+2]

    # relative_scale=188.7
    relative_scale=100
    vertical_offset=-0.6
    out_data[0,:,unity_joints_vals.get_loc("Hips_Xposition")] = smpl_trans[:,0]/relative_scale
    out_data[0,:,unity_joints_vals.get_loc("Hips_Yposition")] = smpl_trans[:,1]/relative_scale+vertical_offset
    out_data[0,:,unity_joints_vals.get_loc("Hips_Zposition")] = smpl_trans[:,2]/relative_scale

    angles2smplbvh(out_data, dest_dir, result_filename, framerate)


