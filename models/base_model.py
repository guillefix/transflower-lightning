import torch
from contextlib import contextmanager
from collections import OrderedDict
print("HOOOOOO")
from pytorch_lightning import LightningModule
print("HOOOOOO")
from .optimizer import get_scheduler, get_optimizers

# Benefits of having one skeleton, e.g. for train - is that you can keep all the incremental changes in
# one single code, making it your streamlined and updated script -- no need to keep separate logs on how
# to implement stuff

class BaseModel(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(vars(opt))
        self.opt = opt
        self.optimizers = []
        self.schedulers = []

    def name(self):
        return 'BaseModel'

    #def setup_opt(self, is_train):
    #    pass

    def configure_optimizers(self):
        self.optimizers = get_optimizers(self, self.opt)
        self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        return self.optimizers, self.schedulers
        #return self.optimizers

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        ABSTRACT METHOD
        :param parser:
        :param is_train:
        :return:
        """
        return parser

