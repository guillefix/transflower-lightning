import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
import glob
import torch
from training.datasets import create_dataset, create_dataloader
from models import create_model
from training.options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup(is_train=True)
    train_dataset = create_dataset(opt)
    train_dataset.setup()
    train_dataloader = create_dataloader(train_dataset)
    if opt.do_validation:
        val_dataset = create_dataset(opt, phase="val")
        val_dataset.setup()
        val_dataloader = create_dataloader(val_dataset)
    print('#training sequences = {:d}'.format(len(train_dataset)))

    default_save_path = opt.checkpoints_dir+"/"+opt.experiment_name

    logger = TensorBoardLogger(opt.checkpoints_dir+"/tb_logs", name=opt.experiment_name)

    if opt.continue_train:
        logs_path = default_save_path+"/lightning_logs"
        checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if os.path.isdir(logs_path+"/"+d)]
        checkpoint_subdirs = sorted(checkpoint_subdirs,key=lambda t: t[1])
        checkpoint_path=logs_path+"/"+checkpoint_subdirs[-1][0]+"/checkpoints/"
        list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        if opt.load_weights_only:
            state_dict = torch.load(latest_file)
            model.load_state_dict(state_dict['state_dict'])
            trainer = Trainer(logger=logger, default_root_dir=default_save_path)
        else:
            trainer = Trainer(logger=logger, default_root_dir=default_save_path, resume_from_checkpoint=latest_file)
    else:
        trainer = Trainer(logger=logger, default_root_dir=default_save_path)

    trainer.fit(model, train_dataloader)

    #for testing..
    #print(trainer.test(model, train_dataloader))
