import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
import glob
import json, yaml
from jsmin import jsmin
from training.datasets import create_dataset, create_dataloader
from models import create_model
from training.options.train_options import TrainOptions
from pytorch_lightning import Trainer
#from test_tube import Experiment
# from pytorch_lightning.callbacks import ModelCheckpoint
# from argparse import ArgumentParser, Namespace

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    train_dataset = create_dataset(opt)
    train_dataset.setup()
    train_dataloader = create_dataloader(train_dataset)
    if opt.val_epoch_freq:
        val_dataset = create_dataset(opt, validation_phase=True)
        val_dataset.setup()
        val_dataloader = create_dataloader(val_dataset)
    print('#training sequences = {:d}'.format(len(train_dataset)))

    default_save_path = opt.checkpoints_dir+"/"+opt.experiment_name

    #checkpoint_callback = ModelCheckpoint(
    #            dirpath=default_save_path,
    #                save_weights_only=True)

    if opt.continue_train:
        logs_path = default_save_path+"/lightning_logs"
        checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if os.path.isdir(logs_path+"/"+d)]
        checkpoint_subdirs = sorted(checkpoint_subdirs,key=lambda t: t[1])

        checkpoint_path=logs_path+"/"+checkpoint_subdirs[-1][0]+"/checkpoints/"
        #checkpoint_path=default_save_path
        list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
        #list_of_files = glob.glob(checkpoint_path+'/epoch*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        if opt.tpu_cores > 0:
            trainer = Trainer(tpu_cores=opt.tpu_cores, max_epochs=10000, default_root_dir=default_save_path, resume_from_checkpoint=latest_file)
        else:
            trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path, resume_from_checkpoint=latest_file)
        #model.load_state_dict(torch.load("./checkpoint.pt"))
        #trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path, checkpoint_callback=checkpoint_callback)
    else:
        if opt.tpu_cores > 0:
            trainer = Trainer(tpu_cores=opt.tpu_cores, max_epochs=10000, default_root_dir=default_save_path)#, checkpoint_callback=checkpoint_callback)
        else:
            trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path)#, checkpoint_callback=checkpoint_callback)

    trainer.fit(model, train_dataloader)
    #print(trainer.test(model, train_dataloader))
