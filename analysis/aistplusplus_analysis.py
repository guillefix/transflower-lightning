import pandas as pd
import numpy as np
import torch
from analysis.aistplusplus_api.convert_mat_to_euler import rot_mats_to_eulers
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib
import pickle

# seqs = [x[:-1].split("_") for x in open("analysis/base_filenames.txt", "r").readlines()]
seqs = [x[:-1] for x in open("analysis/base_filenames.txt", "r").readlines()]

# seqs = [{"genre":x[0], "situation":x[1], "camera":x[2], "dancer":x[3], "musicId":x[4], "choreo":x[5]} for x in seqs]
#
# df = pd.DataFrame(seqs)

data_dir="data/scaled_features/"

# data = np.load(data_dir + seqs[10]+".joint_angles_scaled.npy")
def get_scaling(seq_id):
    smpl_thing = pickle.load(open("../aistpp_data/aist_plusplus_final/motions/"+seq_id+".pkl", "rb"))
    smpl_poses = smpl_thing['smpl_poses']
    smpl_scaling = smpl_thing['smpl_scaling']
    smpl_trans = smpl_thing['smpl_trans']
    return smpl_scaling[0]

max_deriv = lambda seq: np.diff(np.load(data_dir + seq+".joint_angles_scaled.npy")).max()
seqs = sorted(seqs, key=max_deriv)
seqs2 = sorted(seqs, key=lambda seq: get_scaling(seq))

#%%

seqs2[-35]
get_scaling(seqs2[-35])

to_check_coz_big = []
with open("ones_to_check.txt", "a") as f:
    for seq in list(reversed(seqs2))[33:]:
        if get_scaling(seq) > 96:
            to_check_coz_big.append(seq)
            # seq = seq.split("_")
            # seq[2] = "c01"
            # seq = "_".join(seq)
            # f.writelines(seq+"\n")

with open("ones_to_check2.txt", "w") as f:
    for seq in list(reversed(seqs))[14:]:
        if max_deriv(seq) > 20 and seq not in to_check_coz_big:
            seq = seq.split("_")
            seq[2] = "c01"
            seq = "_".join(seq)
            f.writelines(seq+"\n")

np.diff(np.load(data_dir + seqs[-15]+".joint_angles_scaled.npy")).max()

data = np.load(data_dir + seqs[-11]+".joint_angles_scaled.npy")

#%%

# with open("bad_ones.txt", "w") as f:
#     for seq in [x.split("/")[-1][:-4] for x in glob.glob("../aistplusplus_api/visualization/bad/*")]:
#         seq = seq.split("_")
#         seq[2] = "cAll"
#         seq = "_".join(seq)
#         f.writelines(seq+"\n")
#         f.writelines(seq+"\n")

train_ones = [x[:-1] for x in open("analysis/aistpp_base_filenames_train_filtered.txt", "r").readlines()]
bad_ones = [x[:-1] for x in open("bad_ones.txt", "r").readlines()]


with open("analysis/aistpp_base_filenames_train_filtered.txt", "w") as f:
    for line in train_ones:
        if line not in bad_ones:
            f.writelines(line+"\n")

#%%

# np.diff(data).max()


plt.plot(np.diff(data[:900]).max(1))
# plt.plot(np.diff(data).mean(1))

#gHO_sFM_cAll_d20_mHO5_ch13 from 350

#gBR_sFM_cAll_d05_mBR4_ch13 up to 900

#gWA_sBM_cAll_d27_mWA4_ch08 except the end

#%%

# seqs.remove(max_diff_seq)

max_diff = 0
max_diff_seq = ""
for seq in seqs:
    data = np.load(data_dir + seq+".joint_angles_scaled.npy")
    diff = np.diff(data).max()
    if diff > max_diff:
        max_diff = diff
        max_diff_seq = seq

max_diff
max_diff_seq
data = np.load(data_dir + max_diff_seq+".joint_angles_scaled.npy")

#%%
#
# plt.ion()
# plt.show()
# for i in range(data.shape[1]):
#     plt.gca().clear()
#     plt.plot(data[:,i])
#     plt.draw()
#     plt.pause(0.03)
#
# plt.plot(np.diff(data[:,0]))

transform = pickle.load(open(data_dir+"/"+"pkl_joint_angles_mats_scaler"+'.pkl', "rb"))
unscaled_data = transform.inverse_transform(data)

unscaled_data.shape

smpl_thing = rot_mats_to_eulers(np.expand_dims(unscaled_data, 1))
smpl_poses,smpl_scaling,smpl_trans = smpl_thing['smpl_poses'], smpl_thing['smpl_scaling'], smpl_thing['smpl_trans']

#%%

import glob, os
import pickle
# for file in glob.glob("../aistpp_data/aist_plusplus_final/motions/*"):
def get_scaling(seq_id):
    smpl_thing = pickle.load(open("../aistpp_data/aist_plusplus_final/motions/"+seq_id+".pkl", "rb"))
    smpl_poses = smpl_thing['smpl_poses']
    smpl_scaling = smpl_thing['smpl_scaling']
    smpl_trans = smpl_thing['smpl_trans']
    return smpl_scaling[0]

# smpl_thing['smpl_poses'], smpl_thing['smpl_scaling'], smpl_thing['smpl_trans'] =

#%%
from analysis.utils import run_bash_command
from smplx import SMPL
import os

audio_file = "a"
seq_id = "a"
output_folder = "analysis/tmp"
root_dir="analysis/tmp"

def delete_images():
    files = glob.glob(root_dir+'/img/*')
    for f in files:
        os.remove(f)

smpl = SMPL(model_path="../aistplusplus_api", gender='MALE', batch_size=1)
output = smpl.forward(
    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    transl=torch.from_numpy(smpl_trans).float(),
    scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
    )
keypoints3d = output.joints.detach().numpy()
keypoints3d = keypoints3d[:,:24] # the body joints (ignoring the extra head, feet and hand bones added onto it here https://github.com/vchoutas/smplx/blob/7547ee6656b942a68a97604d0cf7b6b834fad9eb/smplx/vertex_joint_selector.py)
# that file takes the position of the vertices corresponding to certain joints
# print(keypoints3d)

# Plot as images
delete_images()
fig = plt.figure()
plt.ion()
plt.show()
ax = Axes3D(fig)
# print(keypoints3d.shape)
# print(keypoints3d[0,:,2])
ax.scatter(keypoints3d[0,:,2], keypoints3d[0,:,0], keypoints3d[0,:,1])
plt.xlim([-100,100])
plt.ylim([-100,100])
ax.set_zlim([75,275])
ax.view_init(0, 0)
plt.draw()
plt.pause(0.001)

for i in range(1,len(keypoints3d)):
    print(i)
    ax.clear()
    ax.scatter(keypoints3d[i,:,2], keypoints3d[i,:,0], keypoints3d[i,:,1])
    plt.xlim([-100,100])
    plt.ylim([-100,100])
    ax.set_zlim([75,275])
    ax.view_init(0, 0)
    plt.draw()
    plt.pause(0.001)
    plt.savefig(root_dir+"/img/img_"+str(i)+".png")

video_file = output_folder+seq_id+".mp4"
video_file2 = output_folder+seq_id+"_music.mp4"
bash_command = "ffmpeg -y -r 60 -f image2 -s 1920x1080 -i "+root_dir+"/img/img_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p "+video_file
run_bash_command(bash_command)
trim_audio=2
if audio_file is not None:
    new_audio_file = output_folder+seq_id+".mp3"
    bash_command = "ffprobe -v 0 -show_entries format=duration -of compact=p=0:nk=1 "+video_file
    duration = float(run_bash_command(bash_command))
    bash_command = "ffmpeg -y -i "+audio_file+" -ss "+str(trim_audio)+" -t "+str(duration)+" "+new_audio_file
    run_bash_command(bash_command)
    bash_command = "ffmpeg -y -i "+video_file+" -i "+new_audio_file+" "+video_file2
    run_bash_command(bash_command)
