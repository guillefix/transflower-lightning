import numpy as np; import scipy.linalg
# LUL
w_shape = [219,219]
w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
np_p, np_l, np_u = scipy.linalg.lu(w_init)

from training.datasets import create_dataset, create_dataloader

from models import create_model
from training.options.train_options import TrainOptions
import pytorch_lightning as pl
import numpy as np
import pickle, json
import sklearn
import argparse
import os, glob

from analysis.visualization.generate_video_from_mats import generate_video_from_mats
from analysis.visualization.generate_video_from_expmaps import generate_video_from_expmaps
from analysis.visualization.generate_video_from_moglow_pos import generate_video_from_moglow_loc

from training.utils import get_latest_checkpoint

if __name__ == '__main__':
    print("Hi")
    parser = argparse.ArgumentParser(description='Generate with model')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seq_id', type=str)
    parser.add_argument('--use_scalers', action='store_true')
    parser.add_argument('--generate_video', action='store_true')
    parser.add_argument('--generate_bvh', action='store_true')
    parser.add_argument('--fps', type=int, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir
    fps = args.fps
    output_folder = args.output_folder
    seq_id = args.seq_id

    if seq_id is None:
        temp_base_filenames = [x[:-1] for x in open(data_dir + "/base_filenames_test.txt", "r").readlines()]
        seq_id = np.random.choice(temp_base_filenames)

    #TODO: change this to load hparams from the particular version folder, that we load the model from, coz the opts could differ between versions potentially.
    exp_opt = json.loads(open("training/experiments/"+args.experiment_name+"/opt.json","r").read())
    opt = vars(TrainOptions().parse(parse_args=["--model", exp_opt["model"]]))
    print(opt)
    opt.update(exp_opt)
    # opt["cond_concat_dims"] = True
    # opt["bn_momentum"] = 0.0
    opt["batch_size"] = 1
    opt["phase"] = "inference"
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    print(opt)
    opt = Struct(**opt)

    input_mods = opt.input_modalities.split(",")
    output_mods = opt.output_modalities.split(",")
    output_time_offsets = [int(x) for x in str(opt.output_time_offsets).split(",")]
    if args.use_scalers:
        scalers = [x+"_scaler.pkl" for x in output_mods]
    else:
        scalers = []

    # Load latest trained checkpoint from experiment
    default_save_path = opt.checkpoints_dir+"/"+opt.experiment_name
    logs_path = default_save_path
    latest_file = get_latest_checkpoint(logs_path)
    print(latest_file)
    model = create_model(opt)
    model = model.load_from_checkpoint(latest_file, opt=opt)

    # Load input features (sequences must have been processed previously into features)
    features = {}
    for mod in input_mods:
        feature = np.load(data_dir+"/"+seq_id+"."+mod+".npy")
        features["in_"+mod] = np.expand_dims(feature,0).transpose((1,0,2))

    # Generate prediction
    model.cuda()
    # import pdb;pdb.set_trace()
    predicted_mods = model.generate(features)
    # import pdb;pdb.set_trace()
    for i, mod in enumerate(output_mods):
        predicted_mod = predicted_mods[i].cpu().numpy()
        if len(scalers)>0:
            transform = pickle.load(open(data_dir+"/"+scalers[i], "rb"))
            predicted_mod = transform.inverse_transform(predicted_mod)
        print(predicted_mod)
        predicted_features_file = output_folder+"/"+args.experiment_name+"/predicted_mods/"+seq_id+"."+mod+".generated"
        np.save(predicted_features_file,predicted_mod)
        predicted_features_file += ".npy"

        if args.generate_video:
            if fps is None:
                fps = 20
            trim_audio = output_time_offsets[i] / fps #converting trim_audio from being in frames (which is more convenient as thats how we specify the output_shift in the models), to seconds
            print("trim_audio: ",trim_audio)

            audio_file = data_dir + "/" + seq_id + ".mp3"

            output_folder = output_folder+"/"+args.experiment_name+"/videos/"

            if mod == "joint_angles_scaled":
                generate_video_from_mats(predicted_features_file,output_folder,audio_file,trim_audio,fps,plot_mats)
            elif mod == "expmap_scaled" or mod == "expmap_20_scaled":
                pipeline_file = f'{data_dir}/motion_{mod}_data_pipe.sav'
                generate_video_from_expmaps(predicted_features_file,pipeline_file,output_folder,audio_file,trim_audio,args.generate_bvh)
            elif mod == "position_scaled":
                control_file = f'{data_dir}/{seq_id}.moglow_control.npy'
                generate_video_from_moglow_loc(predicted_features_file,control_file,output_folder,audio_file,fps,trim_audio)
            else:
                print("Warning: mod "+mod+" not supported")
                # raise NotImplementedError(f'Feature type {feature_type} not implemented')
                pass
