from training.datasets import create_dataset, create_dataloader
from models import create_model
from training.options.train_options import TrainOptions
import pytorch_lightning as pl
import numpy as np
import pickle, json
import sklearn
import argparse
import os, glob

if __name__ == '__main__':
    print("Hi")
    parser = argparse.ArgumentParser(description='Generate dance from song')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seq_id', type=str)
    parser.add_argument('--input_modalities', type=str)
    parser.add_argument('--output_modalities', type=str)
    args = parser.parse_args()
    data_dir = args.data_dir
    seq_id = args.seq_id
    input_mods = args.input_modalities.split(",")
    output_mods = args.output_modalities.split(",")

    opt = json.loads(open("training/experiments/"+args.experiment_name+"/opt.json","r").read())
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    opt = Struct(**opt)

    # Load latest trained checkpoint from experiment
    default_save_path = opt.checkpoints_dir+"/"+opt.experiment_name
    logs_path = default_save_path+"/lightning_logs"
    checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if os.path.isdir(logs_path+"/"+d)]
    checkpoint_subdirs = sorted(checkpoint_subdirs,key=lambda t: t[1])
    checkpoint_path=logs_path+"/"+checkpoint_subdirs[-1][0]+"/checkpoints/"
    list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    model = create_model(opt)
    model = model.load_from_checkpoint(latest_file, opt=opt)

    # Load input features (sequences must have been processed previously into features)
    features = {}
    for mod in input_mods:
        feature = np.load(data_dir+"/"+seq_id+"."+mod+".npy")
        features["in_"+mod] = np.expand_dims(np.expand_dims(feature.transpose(1,0),0),0)

    # Generate prediction
    model.cuda()
    predicted_modes = model.generate(features)[0].cpu().numpy()
    # At the moment we are hardcoding the output mod. TODO: make more general
    transform = pickle.load(open(data_dir+"/"+'pkl_joint_angles_mats'+'_'+'scaler'+'.pkl', "rb"))
    predicted_modes = transform.inverse_transform(predicted_modes)
    print(predicted_modes)
    np.save("generated/"+seq_id+".pkl_joint_angles_mats.generated.test.npz",predicted_modes)
