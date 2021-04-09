import torch
from contextlib import contextmanager
from collections import OrderedDict
print("HOOOOOO")
from pytorch_lightning import LightningModule
print("HOOOOOO")
from .optimizer import get_scheduler, get_optimizers

from models.util.generation import autoregressive_generation_multimodal

# Benefits of having one skeleton, e.g. for train - is that you can keep all the incremental changes in
# one single code, making it your streamlined and updated script -- no need to keep separate logs on how
# to implement stuff

class BaseModel(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(vars(opt))
        self.opt = opt
        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        self.output_lengths = output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]
        self.output_time_offsets = output_time_offsets = [int(x) for x in self.opt.output_time_offsets.split(",")]
        self.input_time_offsets = input_time_offsets = [int(x) for x in self.opt.input_time_offsets.split(",")]
        self.optimizers = []
        self.schedulers = []

    def name(self):
        return 'BaseModel'

    #def setup_opt(self, is_train):
    #    pass

    def configure_optimizers(self):
        optimizers = get_optimizers(self, self.opt)
        schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        return optimizers, schedulers
        #return self.optimizers

    def set_inputs(self, data):
        # BTC -> TBC
        self.inputs = []
        self.targets = []
        for i, mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_ = input_.permute(1,0,2)
            self.inputs.append(input_)
        for i, mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_ = target_.permute(1,0,2)
            self.targets.append(target_)

    def generate(self,features, teacher_forcing=False):
        output_seq = autoregressive_generation_multimodal(features, self, autoreg_mods=self.output_mods, teacher_forcing=teacher_forcing)
        return output_seq

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

