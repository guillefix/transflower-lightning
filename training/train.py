import time
from training.datasets import create_dataset, create_dataloader
from models import create_model
import random
from training.options.train_options import TrainOptions
import pytorch_lightning as pl
from test_tube import Experiment

if __name__ == '__main__':
    print("Hi")
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

    if opt.continue_train:
        model = model.load_from_checkpoint("lightning_logs/version_17/checkpoints/epoch=60-step=243.ckpt", opt=opt)

    trainer = pl.Trainer(num_processes=opt.workers,gpus=opt.gpu_ids)
    trainer.fit(model, train_dataloader)
