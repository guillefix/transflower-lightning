import os
import torch
from torch import nn
import torch.nn.functional as F
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

from .moglow.models import Glow

class TransflowerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        self.douts = douts = [int(x) for x in self.opt.douts.split(",")]
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        self.output_lengths = output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]
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
        self.criterion = nn.MSELoss()

    def name(self):
        return "Transformer"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dins', default=None)
        parser.add_argument('--glow_K', type=int, default=516)
        parser.add_argument('--actnorm_scale', type=float, default=1.0)
        parser.add_argument('--flow_permutation', type=str, default="invconv")
        parser.add_argument('--flow_coupling', type=str, default="affine")
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--network_model', type=str, default="LSTM")
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--LU_decomposed', action='store_true')
        return parser

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        latents = []
        for i, mod in enumerate(self.input_mods):
            mask = getattr(self,"src_mask_"+str(i))
            #mask = self.src_masks[i]
            latents.append(self.input_mod_nets[i].forward(data[i],mask))
        latent = torch.cat(latents)
        outputs = []
        for i, mod in enumerate(self.output_mods):
            mask = getattr(self,"out_mask_"+str(i))
            #mask = self.output_masks[i]
            trans_output = self.output_mod_nets[i].forward(latent,mask)[:self.conditioning_output_lenghts[i]]
            output, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
            outputs.append(output.permute(1,0,2))

        # import pdb;pdb.set_trace()
        #shape

        return outputs

    def generate(self,features):
        opt = self.opt
        inputs_ = []
        for i,mod in enumerate(self.input_mods):
            input_ = features["in_"+mod]
            input_ = torch.from_numpy(input_).float().cuda()
            input_shape = input_.shape
            input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(2,0,1).to(self.device)
            inputs_.append(input_)


        inputs = []
        input_tmp = []
        for i,mod in enumerate(self.input_mods):
            input_tmp.append(inputs_[i].clone()[self.input_time_offsets[i]:self.input_time_offsets[i]+self.input_lengths[i]])

        self.eval()
        output_seq = []
        # sequence_length = inputs_[0].shape[0]
        sequence_length = inputs_[1].shape[0]
        with torch.no_grad():
            # for t in range(min(512, sequence_length-max(self.input_lengths)-1)):
            for t in range(sequence_length-max(self.input_lengths)-1):
                # for t in range(sequence_length):
                print(t)
                inputs = [x.clone().cuda() for x in input_tmp]
                outputs = self.forward(inputs)
                if t == 0:
                    for i, mod in enumerate(self.output_mods):
                        output = outputs[i]
                        # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                        output_seq.append(output[:1].detach().clone())
                        # output_seq.append(inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]+0.15*torch.randn(1,219).cuda())
                else:
                    for i, mod in enumerate(self.output_mods):
                        # output_seq[i] = torch.cat([output_seq[i], inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]+0.15*torch.randn(1,219).cuda()])
                        output = outputs[i]
                        output_seq[i] = torch.cat([output_seq[i], output[:1].detach().clone()])
                        # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                        # print(outputs[i][:1])
                if t < sequence_length-1:
                    for i, mod in enumerate(self.input_mods):
                        if mod in self.output_mods: #TODO: need flag to mark if autoregressive
                            j = self.output_mods.index(mod)
                            output = outputs[i]
                            input_tmp[i] = torch.cat([input_tmp[i][1:],output[:1].detach().clone()],0)
                            # input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]],0)
                            print(torch.mean((inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]-self.predicted_inputs[i]+1:t+self.input_time_offsets[i]+self.input_lengths[i]-self.predicted_inputs[i]+1+1]-outputs[j][:1].detach().clone())**2))
                        else:
                            input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][self.input_time_offsets[i]+self.input_lengths[i]+t:self.input_time_offsets[i]+self.input_lengths[i]+t+1]],0)

            return output_seq

    def set_inputs(self, data):
        self.inputs = []
        self.targets = []
        for i, mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_shape = input_.shape
            if len(input_shape)==4:
                # It's coming as 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
                input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3]))
            input_ = input_.permute(2,0,1)
            self.inputs.append(input_)
        for i, mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_shape = target_.shape
            if len(target_shape)==4:
                target_ = target_.reshape((target_shape[0]*target_shape[1], target_shape[2], target_shape[3]))
            target_ = target_.permute(2,0,1)
            self.targets.append(target_)

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        latents = []
        for i, mod in enumerate(self.input_mods):
            mask = getattr(self,"src_mask_"+str(i))
            latents.append(self.input_mod_nets[i].forward(self.inputs[i],mask))

        latent = torch.cat(latents)
        loss = 0
        for i, mod in enumerate(self.output_mods):
            mask = getattr(self,"out_mask_"+str(i))
            output = self.output_mod_nets[i].forward(latent,mask)[:self.conditioning_output_lenghts[i]]
            glow = self.output_mod_glows[i]
            # import pdb;pdb.set_trace()
            z, sldj = glow(x=self.targets[i].permute(1,0,2), cond=output.permute(1,0,2)) #time, batch, features -> batch, time, features
            #print(sldj)
            loss += glow.loss_generative(z, sldj)
        self.log('nll_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        self.set_inputs(batch)
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.inputs[i],self.src_masks[i]))

        latent = torch.cat(latents)
        loss_mse = 0
        for i, mod in enumerate(self.output_mods):
            output = self.output_mod_nets[i].forward(latent,self.output_masks[i])[:self.output_lengths[i]]
            #print(output)
            loss_mse += self.criterion(output, self.targets[i])
            #loss_mse += self.criterion(output, self.targets[i]).detach()
        print(loss_mse)
        #return loss_mse
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    #to help debug XLA stuff, like missing ops, or data loading/compiling bottlenecks
    # see https://youtu.be/iwtpwQRdb3Y?t=1056
    #def on_epoch_end(self):
    #    xm.master_print(met.metrics_report())


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
