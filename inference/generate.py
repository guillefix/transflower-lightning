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

if __name__ == '__main__':
    print("Hi")
    parser = argparse.ArgumentParser(description='Generate dance from song')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seq_id', type=str)
    parser.add_argument('--input_modalities', type=str)
    parser.add_argument('--output_modalities', type=str)
    parser.add_argument('--scalers', type=str, default="")
    args = parser.parse_args()
    data_dir = args.data_dir
    seq_id = args.seq_id
    input_mods = args.input_modalities.split(",")
    output_mods = args.output_modalities.split(",")
    scalers = [x for x in args.scalers.split(",") if len(x)>0]

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

    # Load latest trained checkpoint from experiment
    default_save_path = opt.checkpoints_dir+"/"+opt.experiment_name
    # logs_path = default_save_path+"/lightning_logs"
    logs_path = default_save_path
    checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if (os.path.isdir(logs_path+"/"+d) and d.split("_")[0]=="version")]
    checkpoint_subdirs = sorted(checkpoint_subdirs,key=lambda t: t[1])
    checkpoint_path=logs_path+"/"+checkpoint_subdirs[-1][0]+"/checkpoints/"
    list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    model = create_model(opt)
    # model.setup(is_train=True)
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
    # At the moment we are hardcoding the output mod. TODO: make more general
    for i, mod in enumerate(output_mods):
        predicted_mod = predicted_mods[i].cpu().numpy()
        if len(scalers)>0:
            transform = pickle.load(open(data_dir+"/"+scalers[i]+'.pkl', "rb"))
            predicted_mod = transform.inverse_transform(predicted_mod)
        print(predicted_mod)
        np.save("inference/generated/"+args.experiment_name+"/predicted_mods/"+seq_id+"."+mod+".generated",predicted_mod)
