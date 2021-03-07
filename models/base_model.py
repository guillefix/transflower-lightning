import torch
from contextlib import contextmanager
from collections import OrderedDict
from pytorch_lightning import LightningModule

# Benefits of having one skeleton, e.g. for train - is that you can keep all the incremental changes in
# one single code, making it your streamlined and updated script -- no need to keep separate logs on how
# to implement stuff

class BaseModel(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def name(self):
        return 'BaseModel'

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

