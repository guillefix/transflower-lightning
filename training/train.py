import time
import os
import glob
from training.datasets import create_dataset, create_dataloader
from models import create_model
import random
from training.options.train_options import TrainOptions
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
#from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint

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
        #model = model.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=287-step=1151.ckpt", opt=opt)

        logs_path = default_save_path+"/lightning_logs"
        checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if os.path.isdir(logs_path+"/"+d)]
        checkpoint_subdirs = sorted(checkpoint_subdirs,key=lambda t: t[1])

        checkpoint_path=logs_path+"/"+checkpoint_subdirs[-1][0]+"/checkpoints/"
        #checkpoint_path=default_save_path
        list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
        #list_of_files = glob.glob(checkpoint_path+'/epoch*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path, resume_from_checkpoint=latest_file)
        #model.load_state_dict(torch.load("./checkpoint.pt"))
        #model = model.load_from_checkpoint(latest_file, opt=opt)
        #trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path)
        #trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path, checkpoint_callback=checkpoint_callback)
        #trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path, checkpoint_callback=None)
        #trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path)
    else:
        trainer = Trainer(num_processes=opt.workers, gpus=opt.gpu_ids, max_epochs=10000, default_root_dir=default_save_path)#, checkpoint_callback=checkpoint_callback)

    trainer.fit(model, train_dataloader)
    #print(trainer.test(model, train_dataloader))
    #torch.save(model.state_dict(), "./checkpoint.pt")
