import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
import glob
import torch
print(torch.cuda.is_available())
from training.datasets import create_dataset, create_dataloader
print("HIII")
from models import create_model
import pytorch_lightning as pl
from training.options.train_options import TrainOptions
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
print("HIII")
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.plugins.training_type.deepspeed import DeepSpeedPlugin
from pytorch_lightning.callbacks import ModelCheckpoint


from training.utils import get_latest_checkpoint

if __name__ == '__main__':
    pl.seed_everything(69420)
    opt = TrainOptions().parse()
    print("loaded options")
    print(opt.experiment_name)
    model = create_model(opt)
    print("loaded model")
    if "tpu_cores" in vars(opt) and opt.tpu_cores is not None and opt.tpu_cores > 0:
        plugins = None
    elif opt.plugins is None:
        print("DDPPlugin")
        plugins = DDPPlugin(find_unused_parameters=opt.find_unused_parameters, num_nodes=opt.num_nodes)
    elif opt.plugins == "deepspeed":
        deepspeed_config = {
                "zero_optimization": {
                    "stage": 2,
                    "cpu_offload":False,
                },    
                #'train_batch_size': opt.batch_size,
                'gradient_clipping': opt.gradient_clip_val,
                'fp16': {
                    'enabled': opt.precision == 16,
                    'loss_scale': 0,
                    'initial_scale_power': 15,
                },
            }
        plugins = DeepSpeedPlugin(config=deepspeed_config)
    else:
        #ddpplugin = DDPPlugin(find_unused_parameters=opt.find_unused_parameters, num_nodes=opt.num_nodes)
        #plugins = [ddpplugin, opt.plugins]
        plugins = opt.plugins

    ##Datasets and dataloaders
    train_dataset = create_dataset(opt)
    train_dataset.setup()
    train_dataloader = create_dataloader(train_dataset)
    if opt.do_validation:
        val_dataset = create_dataset(opt, split="val")
        val_dataset.setup()
        val_dataloader = create_dataloader(val_dataset, split="val")
    if opt.do_testing:
        test_dataset = create_dataset(opt, split="test")
        test_dataset.setup()
        test_dataloader = create_dataloader(test_dataset, split="test")
    print('#training sequences = {:d}'.format(len(train_dataset)))

    default_save_path = opt.checkpoints_dir+"/"+opt.experiment_name

    logger = TensorBoardLogger(opt.checkpoints_dir, name=opt.experiment_name)
    checkpoint_callback = ModelCheckpoint(
            monitor = 'loss',
            save_top_k = 5
            )
    callbacks = [checkpoint_callback]
    args = Trainer.parse_argparser(opt)

    if opt.continue_train:
        print("CONTINUE TRAIN")
        logs_path = default_save_path
        latest_file = get_latest_checkpoint(logs_path)
        print(latest_file)
        if opt.load_weights_only:
            state_dict = torch.load(latest_file)
            state_dict = state_dict['state_dict']
            load_strict = True
            if opt.only_load_in_state_dict != "":
                state_dict = {k:v for k,v in state_dict.items() if (opt.only_load_in_state_dict in k)}
                load_strict = False
            if opt.ignore_in_state_dict != "":
                state_dict = {k:v for k,v in state_dict.items() if not (opt.ignore_in_state_dict in k)}
                load_strict = False
            model.load_state_dict(state_dict, strict=load_strict)
            trainer = Trainer.from_argparse_args(args, logger=logger, default_root_dir=default_save_path, plugins=plugins, callbacks=callbacks)
        else:
            trainer = Trainer.from_argparse_args(args, logger=logger, default_root_dir=default_save_path, resume_from_checkpoint=latest_file, plugins=plugins, callbacks=callbacks)
    else:
        trainer = Trainer.from_argparse_args(args, logger=logger, default_root_dir=default_save_path, plugins=plugins, callbacks=callbacks)

    #Tuning
    if opt.do_tuning:
        if opt.do_validation:
            trainer.tune(model, train_dataloader, val_dataloader)
        else:
            trainer.tune(model, train_dataloader)

    #Training
    if not opt.skip_training:
        if opt.do_validation:
            trainer.fit(model, train_dataloader, val_dataloader)
        else:
            trainer.fit(model, train_dataloader)

    #evaluating on test set
    if opt.do_testing:
        print("TESTING")
        logs_path = default_save_path
        latest_file = get_latest_checkpoint(logs_path)
        print(latest_file)
        state_dict = torch.load(latest_file)
        model.load_state_dict(state_dict['state_dict'])
        trainer.test(model, test_dataloader)

        # trainer = Trainer(logger=logger)
        # # trainer.test(model, train_dataloader)
        # logs_path = default_save_path
        # checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if os.path.isdir(logs_path+"/"+d)]
        # checkpoint_subdirs = sorted(checkpoint_subdirs,key=lambda t: t[1])
        # checkpoint_path=logs_path+"/"+checkpoint_subdirs[-1][0]+"/checkpoints/"
        # list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
        # latest_file = max(list_of_files, key=os.path.getctime)
        # print(latest_file)
        # trainer.test(model, test_dataloaders=test_dataloader, ckpt_path=latest_file)
        # trainer.test(test_dataloaders=test_dataloader, ckpt_path=latest_file)
        # trainer.test(test_dataloaders=test_dataloader)
