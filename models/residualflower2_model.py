import torch
from torch import nn
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast

from .util.generation import autoregressive_generation_multimodal
import argparse
from argparse import Namespace
import models

class Residualflower2Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        self.output_lengths = output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]
        self.output_time_offsets = output_time_offsets = [int(x) for x in self.opt.output_time_offsets.split(",")]
        self.input_time_offsets = input_time_offsets = [int(x) for x in self.opt.input_time_offsets.split(",")]

        opt_vars = vars(opt)
        mean_vars = self.get_argvars(opt.mean_model, opt)
        mean_opt = opt_vars.copy()
        for k,v in mean_vars.items():
            val = mean_opt["mean_"+k]
            if k not in mean_opt:
                mean_opt[k] = val
            del mean_opt["mean_"+k]
        mean_opt = Namespace(**mean_opt)
        self.mean_model = models.create_model_by_name(opt.mean_model, mean_opt)

        residual_vars = self.get_argvars(opt.residual_model, opt)
        residual_opt = opt_vars.copy()
        for k,v in residual_vars.items():
            val = residual_opt["residual_"+k]
            if k not in residual_opt:
                residual_opt[k] = val
            del residual_opt["residual_"+k]
        residual_opt = Namespace(**residual_opt)
        self.residual_model = models.create_model_by_name(opt.residual_model, residual_opt)

        self.mean_loss = nn.MSELoss()

    def name(self):
        return "Transflower"

    @staticmethod
    def get_argvars(model_name, opt):
        temp_parser = argparse.ArgumentParser()
        model_option_setter = models.get_option_setter(model_name)
        vs = vars(model_option_setter(temp_parser, opt).parse_args([]))
        return vs

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dins', default=None)
        parser.add_argument('--douts', default=None)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--mean_model', type=str, default="transformer")
        parser.add_argument('--residual_model', type=str, default="transflower")
        opt, _ = parser.parse_known_args()
        mean_vars = Residualflower2Model.get_argvars(opt.mean_model, opt)
        for k,v in mean_vars.items():
            print(k)
            if type(v) != type(True):
                if type(v) != type(None):
                    parser.add_argument('--mean_'+k, type=type(v), default=v)
                else:
                    parser.add_argument('--mean_'+k, default=v)
            else:
                parser.add_argument('--mean_'+k, action="store_true")
        residual_vars = Residualflower2Model.get_argvars(opt.residual_model, opt)
        for k,v in residual_vars.items():
            if type(v) != type(True):
                if type(v) != type(None):
                    parser.add_argument('--residual_'+k, type=type(v), default=v)
                else:
                    parser.add_argument('--residual_'+k, default=v)
            else:
                parser.add_argument('--residual_'+k, action="store_true")
        return parser

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        predicted_means = self.mean_model(data)
        predicted_residuals = self.residual_model(data)
        outputs = []
        for i, mod in enumerate(self.output_mods):
            outputs.append(predicted_means[i]+predicted_residuals[i])
        return outputs

    def generate(self,features, teacher_forcing=False):
        inputs_ = []
        for i,mod in enumerate(self.input_mods):
            input_ = features["in_"+mod]
            input_ = torch.from_numpy(input_).float().cuda()
            input_shape = input_.shape
            input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(2,0,1).to(self.device)
            inputs_.append(input_)
        output_seq = autoregressive_generation_multimodal(inputs_, self, autoreg_mods=self.output_mods, teacher_forcing=teacher_forcing)
        return output_seq

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)

        self.mean_model.set_inputs(batch)
        predicted_means = self.mean_model(self.inputs)
        mse_loss = 0
        for i, mod in enumerate(self.output_mods):
            mse_loss += 100*self.mean_loss(predicted_means[i], self.targets[i])

        for i, mod in enumerate(self.output_mods):
            # import pdb;pdb.set_trace()
            batch["out_"+mod] = batch["out_"+mod] - predicted_means[i].permute(1,0,2)

        # self.residual_model.set_inputs(batch)
        nll_loss = self.residual_model.training_step(batch, batch_idx)
        loss = mse_loss + nll_loss
        print("mse_loss: ", mse_loss)
        print("nll_loss: ", nll_loss)
        self.log('mse_loss', mse_loss)
        self.log('nll_loss', nll_loss)
        self.log('loss', loss)
        return loss
