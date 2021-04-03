import torch
from torch import nn
from models import BaseModel
from .util.generation import autoregressive_generation_multimodal
from .moglow.models import Glow

class MoglowModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        # self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        # self.output_lengths = output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]
        self.output_time_offsets = output_time_offsets = [int(x) for x in self.opt.output_time_offsets.split(",")]
        self.input_time_offsets = input_time_offsets = [int(x) for x in self.opt.input_time_offsets.split(",")]

        if len(output_time_offsets) < len(output_mods):
            if len(output_time_offsets) == 1:
                self.output_time_offsets = output_time_offsets = output_time_offsets*len(output_mods)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        if len(input_time_offsets) < len(input_mods):
            if len(input_time_offsets) == 1:
                self.input_time_offsets = input_time_offsets = input_time_offsets*len(input_mods)
            else:
                raise Exception("number of input_time_offsets doesnt match number of input_mods")

        # import pdb;pdb.set_trace()
        glow = Glow(dins[0], dins[1], self.opt)
        setattr(self, "net"+"_glow", glow)

        self.inputs = []
        self.targets = []
        # self.criterion = nn.MSELoss()

    def name(self):
        return "Moglow"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dins', default=None)
        parser.add_argument('--glow_K', type=int, default=16)
        parser.add_argument('--actnorm_scale', type=float, default=1.0)
        parser.add_argument('--flow_permutation', type=str, default="invconv")
        parser.add_argument('--flow_coupling', type=str, default="affine")
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--network_model', type=str, default="LSTM")
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--LU_decomposed', action='store_true')
        return parser

    def forward(self, data, eps_std=1.0):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.net_glow(z=None, cond=data, eps_std=eps_std, reverse=True)
        return outputs

    def generate(self,features, teacher_forcing=False):
        output_seq = autoregressive_generation_multimodal(features, self, autoreg_mods=self.output_mods, teacher_forcing=teacher_forcing)
        return output_seq

    def on_train_start(self):
        self.net_glow.init_lstm_hidden()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # self.zero_grad()
        self.net_glow.init_lstm_hidden()

    def set_inputs(self, data):
        self.inputs = []
        self.targets = []
        for i, mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_shape = input_.shape
            if len(input_shape)==4:
                # It's coming as 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
                input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3]))
            # input_ = input_.permute(2,0,1)
            self.inputs.append(input_)
        for i, mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_shape = target_.shape
            if len(target_shape)==4:
                target_ = target_.reshape((target_shape[0]*target_shape[1], target_shape[2], target_shape[3]))
            # target_ = target_.permute(2,0,1)
            self.targets.append(target_)

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        z, nll = self.net_glow(x=self.inputs[0], cond=self.inputs[1])

        loss = Glow.loss_generative(nll)
        self.log('nll_loss', loss)
        # import pdb;pdb.set_trace()
        return loss

    #to help debug XLA stuff, like missing ops, or data loading/compiling bottlenecks
    # see https://youtu.be/iwtpwQRdb3Y?t=1056
    #def on_epoch_end(self):
    #    xm.master_print(met.metrics_report())


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
