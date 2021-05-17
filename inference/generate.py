import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
import numpy as np; import scipy.linalg
# LUL
w_shape = [219,219]
w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
np_p, np_l, np_u = scipy.linalg.lu(w_init)

from training.datasets import create_dataset, create_dataloader

from models import create_model
from training.options.train_options import TrainOptions
import torch
import pytorch_lightning as pl
import numpy as np
import pickle, json, yaml
import sklearn
import argparse
import os, glob
from pathlib import Path

from analysis.visualization.generate_video_from_mats import generate_video_from_mats
from analysis.visualization.generate_video_from_expmaps import generate_video_from_expmaps
from analysis.visualization.generate_video_from_moglow_pos import generate_video_from_moglow_loc

from training.utils import get_latest_checkpoint

if __name__ == '__main__':
    print("Hi")
    parser = argparse.ArgumentParser(description='Generate with model')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--audio_format', type=str, default="wav")
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seq_id', type=str)
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--no-use_scalers', dest='use_scalers', action='store_false')
    parser.add_argument('--generate_video', action='store_true')
    parser.add_argument('--generate_bvh', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    data_dir = args.data_dir
    audio_format = args.audio_format
    fps = args.fps
    output_folder = args.output_folder
    seq_id = args.seq_id

    if seq_id is None:
        temp_base_filenames = [x[:-1] for x in open(data_dir + "/base_filenames_test.txt", "r").readlines()]
        seq_id = np.random.choice(temp_base_filenames)

    print(seq_id)

    #load hparams file
    default_save_path = "training/experiments/"+args.experiment_name
    logs_path = default_save_path
    latest_checkpoint = get_latest_checkpoint(logs_path)
    print(latest_checkpoint)
    checkpoint_dir = Path(latest_checkpoint).parent.parent.absolute()
    # exp_opt = json.loads(open("training/experiments/"+args.experiment_name+"/opt.json","r").read())
    exp_opt = yaml.load(open(str(checkpoint_dir)+"/hparams.yaml","r").read())
    opt = vars(TrainOptions().parse(parse_args=["--model", exp_opt["model"]]))
    print(opt)
    opt.update(exp_opt)
    # opt["cond_concat_dims"] = True
    # opt["bn_momentum"] = 0.0
    opt["batch_size"] = 1
    opt["phase"] = "inference"
    opt["tpu_cores"] = 0
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
    model = create_model(opt)
    model = model.load_from_checkpoint(latest_checkpoint, opt=opt)

    # Load input features (sequences must have been processed previously into features)
    features = {}
    for mod in input_mods:
        feature = np.load(data_dir+"/"+seq_id+"."+mod+".npy")
        if args.max_length != -1:
            feature = feature[:args.max_lengt]
        features["in_"+mod] = np.expand_dims(feature,0).transpose((1,0,2))

    # Generate prediction
    if torch.cuda.is_available():
        model.cuda()
    #import pdb;pdb.set_trace()
    #import time
    #start_time = time.time()
    predicted_mods = model.generate(features)
    #print("--- %s seconds ---" % (time.time() - start_time))
    if len(predicted_mods) == 0:
        print("Sequence too short!")
    else:
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
                trim_audio = output_time_offsets[i] / fps #converting trim_audio from being in frames (which is more convenient as thats how we specify the output_shift in the models), to seconds
                print("trim_audio: ",trim_audio)

                audio_file = data_dir + "/" + seq_id + "."+audio_format

                output_folder = output_folder+"/"+args.experiment_name+"/videos/"

                if mod == "joint_angles_scaled":
                    generate_video_from_mats(predicted_features_file,output_folder,audio_file,trim_audio,fps,plot_mats)
                elif mod == "expmap_scaled" or mod == "expmap_scaled_20" or mod == "expmap_cr_scaled_20":
                    pipeline_file = f'{data_dir}/motion_{mod}_data_pipe.sav'
                    generate_video_from_expmaps(predicted_features_file,pipeline_file,output_folder,audio_file,trim_audio,args.generate_bvh)
                elif mod == "position_scaled":
                    control_file = f'{data_dir}/{seq_id}.moglow_control_scaled.npy'
                    data = np.load(predicted_features_file)[:,0,:]
                    control = np.load(control_file)
                    if args.use_scalers:
                        transform = pickle.load(open(data_dir+"/moglow_control_scaled_scaler.pkl", "rb"))
                        control = transform.inverse_transform(control)
                    control = control[int(opt.output_time_offsets.split(",")[0]):]
                    generate_video_from_moglow_loc(data,control,output_folder,seq_id,audio_file,fps,trim_audio)
                else:
                    print("Warning: mod "+mod+" not supported")
                    # raise NotImplementedError(f'Feature type {feature_type} not implemented')
                    pass
