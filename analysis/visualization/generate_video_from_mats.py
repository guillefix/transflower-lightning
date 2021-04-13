from smplx import SMPL
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
root_dir = os.path.dirname(os.path.realpath(__file__))
from analysis.visualization.convert_mat_to_euler import rot_mats_to_eulers
from .utils import generate_video_from_images, join_video_and_audio

def delete_images():
    files = glob.glob(root_dir+'/img/*')
    for f in files:
        os.remove(f)

def generate_video_from_mats(pred_mats_file, output_folder, audio_file, trim_audio=0, fps=60, plot_mats=False):
    filename = os.path.basename(pred_mats_file)
    seq_id = filename.split(".")[0]
    predicted_mats = np.load(pred_mats_file)
    if plot_mats:
        plt.matshow(predicted_mats[:1000,0,:-3])
        plt.savefig(root_dir+"/last.predicted_mats.png")

    smpl_thing = rot_mats_to_eulers(predicted_mats)
    smpl_poses,smpl_scaling,smpl_trans = smpl_thing['smpl_poses'], smpl_thing['smpl_scaling'], smpl_thing['smpl_trans']
    #this loads the SMPL_MALE pickle
    smpl = SMPL(model_path="./aistplusplus_api", gender='MALE', batch_size=1) #what if we try gender='FEMALE'?
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
    generate_video_from_images(root_dir+"/img", video_file, fps)

    if audio_file is not None:
        join_video_and_audio(video_file, audio_file, trim_audio)
