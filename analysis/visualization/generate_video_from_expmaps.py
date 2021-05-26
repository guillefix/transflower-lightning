import pickle
import matplotlib.pyplot as plt
import numpy as np
from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline
import joblib as jl
from .utils import generate_video_from_images, join_video_and_audio

import matplotlib
matplotlib.use("Agg")

def generate_video_from_expmaps(features_file, pipeline_file, output_folder, audio_file, trim_audio=0, generate_bvh=False):
    data = np.load(features_file)
    # pipeline = jl.load("data/scaled_features/motion_data_pipe.sav")
    # containing_path = os.path.dirname(features_file)
    # pipeline_file = containing_path + "/" + "motion_expmap_data_pipe.sav"
    pipeline = jl.load(pipeline_file)

    filename = os.path.basename(features_file)
    seq_id = filename.split(".")[0]

    bvh_data=pipeline.inverse_transform([data[:,0,:]])
    if generate_bvh:
        writer = BVHWriter()
        with open(output_folder+"/"+seq_id+".bvh",'w') as f:
            writer.write(bvh_data[0], f)

    bvh2pos = MocapParameterizer('position')
    pos_data = bvh2pos.fit_transform(bvh_data)
    video_file = f'{output_folder}/{seq_id}.mp4'
    #render_mp4(pos_data[0], video_file, axis_scale=100, elev=45, azim=45)
    render_mp4(pos_data[0], video_file, axis_scale=300, elev=45, azim=45)
    if audio_file is not None:
        join_video_and_audio(video_file, audio_file, trim_audio)

