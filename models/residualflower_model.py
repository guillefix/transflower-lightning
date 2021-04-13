import torch
from torch import nn
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast

from .util.generation import autoregressive_generation_multimodal
from .transformer_model import TransformerModel

#TODO: refactor a whole bunch of stuff

class ResidualflowerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        input_mods = self.input_mods
        output_mods = self.output_mods
        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        dins = self.dins
        douts = self.douts
        if self.opt.conditioning_seq_lens is not None:
            self.conditioning_seq_lens = [int(x) for x in str(self.opt.conditioning_seq_lens).split(",")]
        else:
            self.conditioning_seq_lens = [int(x) for x in str(self.opt.output_lengths).split(",")]

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.output_mod_glows = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device, use_pos_emb=True, input_length=input_lengths[i]).to(self.device)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):
            if self.opt.cond_concat_dims:
                net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths)).to(self.device)
            else:
                net = BasicTransformerModel(douts[i]//2, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths)).to(self.device)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)

            # import pdb;pdb.set_trace()
            glow = FlowPlusPlus(scales=ast.literal_eval(opt.scales),
                                in_shape=(douts[i], output_lengths[i], 1),
                                cond_dim=opt.dhid,
                                mid_channels=opt.dhid_flow,
                                num_blocks=opt.num_glow_coupling_blocks,
                                num_components=opt.num_mixture_components,
                                use_attn=opt.glow_use_attn,
                                use_logmix=opt.num_mixture_components>0,
                                drop_prob=opt.dropout,
                                num_heads=opt.num_heads_flow,
                                use_transformer_nn=opt.use_transformer_nn,
                                use_pos_emb=opt.use_pos_emb_coupling,
                                norm_layer = opt.glow_norm_layer,
                                bn_momentum = opt.glow_bn_momentum,
                                cond_concat_dims=opt.cond_concat_dims,
                                cond_seq_len=self.conditioning_seq_lens[i],
                                )
            name = "_output_glow_"+mod
            setattr(self, "net"+name, glow)
            self.output_mod_glows.append(glow)

        self.mean_model = TransformerModel(opt)

        #This is feature creep. Will remove soon
        self.inputs = []
        self.targets = []
        self.criterion = nn.MSELoss()

    def name(self):
        return "Transflower"

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dhid_flow', type=int, default=512)
        parser.add_argument('--predicted_inputs', default="0")
        parser.add_argument('--conditioning_seq_lens', type=str, default=None, help="the number of outputs of the conditioning transformers to feed (meaning the number of elements along the sequence dimension)")
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--num_heads_flow', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--scales', type=str, default="[[10,0]]")
        parser.add_argument('--glow_norm_layer', type=str, default=None)
        parser.add_argument('--glow_bn_momentum', type=float, default=0.1)
        parser.add_argument('--num_glow_coupling_blocks', type=int, default=10)
        parser.add_argument('--num_mixture_components', type=int, default=0)
        parser.add_argument('--glow_use_attn', action='store_true', help="whether to use the internal attention for the FlowPlusPLus model")
        parser.add_argument('--use_transformer_nn', action='store_true', help="whether to use the internal attention for the FlowPlusPLus model")
        parser.add_argument('--use_pos_emb_output', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--use_pos_emb_coupling', action='store_true', help="whether to use positional embeddings for the coupling layer transformers")
        parser.add_argument('--cond_concat_dims', action='store_true', help="if set we concatenate along the channel dimension with with the x for the coupling layer; otherwise we concatenate along the sequence dimesion")
        return parser

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        predicted_means = self.mean_model(data)
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(data[i]))
        latent = torch.cat(latents)
        outputs = []
        for i, mod in enumerate(self.output_mods):
            trans_output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
            output, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
            outputs.append(predicted_means[i]+output.permute(1,0,2))

        return outputs

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        predicted_means = self.mean_model(self.inputs)
        mse_loss = 0
        for i, mod in enumerate(self.output_mods):
            mse_loss += 100*self.criterion(predicted_means[i], self.targets[i])
        #print("mse_loss: ", mse_loss)
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.inputs[i]))

        latent = torch.cat(latents)
        nll_loss=0
        for i, mod in enumerate(self.output_mods):
            output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
            glow = self.output_mod_glows[i]
            # import pdb;pdb.set_trace()
            z, sldj = glow(x=self.targets[i].permute(1,0,2)-predicted_means[i].detach().permute(1,0,2), cond=output.permute(1,0,2)) #time, batch, features -> batch, time, features
            #print(sldj)
            n_timesteps = self.targets[i].shape[1]
            nll_loss += glow.loss_generative(z, sldj)

        loss = mse_loss + nll_loss
        #print("nll_loss: ", nll_loss)
        self.log('mse_loss', mse_loss)
        self.log('nll_loss', nll_loss)
        self.log('loss', loss)
        return loss

    #to help debug XLA stuff, like missing ops, or data loading/compiling bottlenecks
    # see https://youtu.be/iwtpwQRdb3Y?t=1056
    # def on_epoch_end(self):
    #    import torch_xla.core.xla_model as xm
    #    import torch_xla.debug.metrics as met
    #    xm.master_print(met.metrics_report())


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
