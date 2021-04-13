import argparse

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(ROOT_DIR)

from analysis.visualization.generate_video_from_mats import generate_video_from_mats
from analysis.visualization.generate_video_from_expmaps import generate_video_from_expmaps
from analysis.visualization.generate_video_from_moglow_pos import generate_video_from_moglow_loc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate video from expmaps')
    parser.add_argument('--feature_type', type=str, help="rot_mat, expmap, position")
    parser.add_argument('--features_file', type=str)
    parser.add_argument('--output_folder', type=str, default="generated/videos/")
    parser.add_argument('--audio_file', type=str, default=None)
    parser.add_argument('--trim_audio', type=float, default=0, help="in frames")
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--plot_mats', action="store_true")
    parser.add_argument('--pipeline_file', type=str)
    parser.add_argument('--control_file', type=str)
    parser.add_argument('--generate_bvh', action="store_true")
    args = parser.parse_args()
    globals().update(vars(args))

    trim_audio /= fps #converting trim_audio from being in frames (which is more convenient as thats how we specify the output_shift in the models), to seconds

    print("trim_audio: ",trim_audio)

    if feature_type == "rot_mat":
        generate_video_from_mats(features_file,output_folder,audio_file,trim_audio,fps,plot_mats)
    elif feature_type == "expmap" or feature_type == "expmap_20":
        assert pipeline_file is not None #Need to supply pipeline file to process exmaps
        generate_video_from_expmaps(features_file,pipeline_file,output_folder,audio_file,trim_audio,generate_bvh)
    elif feature_type == "moglow_loc":
        assert control_file is not None
        generate_video_from_moglow_loc(features_file,control_file,output_folder,audio_file,fps,trim_audio)
    else:
        raise NotImplementedError(f'Feature type {feature_type} not implemented')
