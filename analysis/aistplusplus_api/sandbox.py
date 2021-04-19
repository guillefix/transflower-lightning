import pickle

thing = pickle.load(open("SMPL_MALE.pkl","rb"), encoding="latin1")

thing.keys()

thing["kintree_table"][0].shape
thing["kintree_table"][0]
thing["J_regressor"].shape
thing["J"].shape
thing["v_template"].shape

from aist_plusplus.loader import AISTDataset
from smplx import SMPL
import torch

#%%

# smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
#     "../multimodal-transformer/data/motions", "gWA_sFM_cAll_d26_mWA1_ch09")
smpl_thing = pickle.load(open("last.generated.test.pkl", "rb"))
smpl_poses,smpl_scaling,smpl_trans = smpl_thing['smpl_poses'], smpl_thing['smpl_scaling'], smpl_thing['smpl_trans']
#MY PICKLE IS PROBABLY WRONG
smpl = SMPL(model_path="./", gender='MALE', batch_size=1)
output = smpl.forward(
    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    transl=torch.from_numpy(smpl_trans).float(),
    scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
    )
keypoints3d = output.joints.detach().numpy()

output.vertices.shape

smpl_poses.shape
keypoints3d.shape

keypoints3d = keypoints3d[:,:24] # the body joints (ignoring the head, feet and hand bones added onto it here https://github.com/vchoutas/smplx/blob/7547ee6656b942a68a97604d0cf7b6b834fad9eb/smplx/vertex_joint_selector.py)
# that file takes the position of the vertices corresponding to certain joints

#%%


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

%matplotlib


import time

from celluloid import Camera
fig = plt.figure()
ax = Axes3D(fig)

# camera = Camera(fig)

import numpy as np
np.max(keypoints3d)

ax.scatter(keypoints3d[0,:,2], keypoints3d[0,:,0], keypoints3d[0,:,1])
plt.show()
plt.xlim([-200,200])
plt.ylim([-200,200])
ax.set_zlim([75,475])
ax.view_init(0, 0)
plt.draw()
#%%
# plt.zlim([-50,50])
for i in range(len(keypoints3d)):
# for i in range(512):
    ax.clear()
    ax.scatter(keypoints3d[i,:,2], keypoints3d[i,:,0], keypoints3d[i,:,1])
    plt.xlim([-100,100])
    plt.ylim([-100,100])
    ax.set_zlim([75,275])
    ax.view_init(0, 0)
    plt.draw()
    plt.savefig("img/img_"+str(i)+".png")
    # camera.snap()

# i=300
# plt.gca().clear()
# plt.scatter(keypoints3d[i,:,0], keypoints3d[i,:,1], keypoints3d[i,:,2])
# plt.show()
#
# a = camera.animate()
# a.save("out.mp4")
# thing['kintree_table'][0]

1
