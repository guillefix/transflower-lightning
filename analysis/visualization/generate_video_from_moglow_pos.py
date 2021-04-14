import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import joblib as jl
from .utils import generate_video_from_images, join_video_and_audio
import os
root_dir = os.path.dirname(os.path.realpath(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

def pre_process_motion_data(clip, init_rot=None, init_trans=None, fps=30, axis_scale=50, elev=45, azim=45):
    rot = init_rot if init_rot is not None else R.from_quat([0,0,0,1])
    translation = init_trans if init_trans is not None else np.array([[0,0,0]])
    translations = np.zeros((clip.shape[0],3))

    joints, root_dx, root_dz, root_dr = clip[:,:-3], clip[:,-3], clip[:,-2], clip[:,-1]
    joints = joints.reshape((len(joints), -1, 3))
    for i in range(len(joints)):
        joints[i,:,:] = rot.apply(joints[i])
        joints[i,:,0] = joints[i,:,0] + translation[0,0]
        joints[i,:,2] = joints[i,:,2] + translation[0,2]
        rot = R.from_rotvec(np.array([0,-root_dr[i],0])) * rot
        translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
        translations[i,:] = translation

    return joints, translation, rot, translations

def generate_video_from_moglow_loc(data, control, output_folder, audio_file, fps, trim_audio=0):
    # import pdb;pdb.set_trace()
    clip = np.concatenate([data,control], axis=1)

    joints, translation, rot, translations = pre_process_motion_data(clip, fps=20, axis_scale=50, elev=45, azim=45)

    fig = plt.figure()
    plt.ion()
    plt.show()
    ax = Axes3D(fig)
    keypoints3d = joints

    ax.scatter(keypoints3d[0,:,0],keypoints3d[0,:,1],keypoints3d[0,:,2])

    plt.xlim([-10,10])
    plt.ylim([0,20])
    ax.set_zlim([-10,10])
    ax.view_init(90,-90)
    plt.draw()
    plt.pause(0.001)
    plt.savefig(root_dir+"/img/img_"+str(0)+".png")

    for i in range(1,len(keypoints3d)):
        print(i)
        ax.clear()
        ax.scatter(keypoints3d[i,:,0], keypoints3d[i,:,1], keypoints3d[i,:,2])
        plt.xlim([-10,10])
        plt.ylim([0,20])
        ax.set_zlim([-10,10])
        ax.view_init(90,-90)
        plt.draw()
        plt.pause(0.001)
        plt.savefig(root_dir+"/img/img_"+str(i)+".png")

    video_file = f'{output_folder}/{seq_id}.mp4'
    generate_video_from_images(root_dir+"/img", video_file, fps)
    if audio_file is not None:
        join_video_and_audio(video_file, audio_file, trim_audio)

