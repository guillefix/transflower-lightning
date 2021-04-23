import torch
from torch import nn
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast
from .util.generation import autoregressive_generation_multimodal

class TransFlowppModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_lengths = self.input_lengths
        output_lengths = self.output_lengths

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
            net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=True, input_length=sum(input_lengths)).to(self.device)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)

            # import pdb;pdb.set_trace()
            glow = FlowPlusPlus(scales=ast.literal_eval(opt.scales),
                                     in_shape=(douts[i], output_lengths[i], 1),
                                     cond_dim=opt.dhid,
                                     mid_channels=opt.dhid,
                                     num_blocks=opt.num_glow_coupling_blocks,
                                     num_components=opt.num_mixture_components,
                                     use_attn=opt.glow_use_attn,
                                     use_logmix=opt.num_mixture_components>0,
                                     drop_prob=opt.dropout
                                     )
            name = "_output_glow_"+mod
            setattr(self, "net"+name, glow)
            self.output_mod_glows.append(glow)


        # self.generate_full_masks()
        self.inputs = []
        self.targets = []
        self.criterion = nn.MSELoss()

    def name(self):
        return "Transformerflow"

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--scales', type=str, default="[[10,0]]")
        parser.add_argument('--num_glow_coupling_blocks', type=int, default=10)
        parser.add_argument('--num_mixture_components', type=int, default=0)
        parser.add_argument('--glow_use_attn', action='store_true', help="whether to use the internal attention for the FlowPlusPLus model")
        return parser

    # def generate_full_masks(self):
    #     input_mods = self.input_mods
    #     output_mods = self.output_mods
    #     input_lengths = self.input_lengths
    #     self.src_masks = []
    #     for i, mod in enumerate(input_mods):
    #         mask = torch.zeros(input_lengths[i],input_lengths[i])
    #         self.register_buffer('src_mask_'+str(i), mask)
    #         self.src_masks.append(mask)
    #
    #     self.output_masks = []
    #     for i, mod in enumerate(output_mods):
    #         mask = torch.zeros(sum(input_lengths),sum(input_lengths))
    #         self.register_buffer('out_mask_'+str(i), mask)
    #         self.output_masks.append(mask)

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        latents = []
        for i, mod in enumerate(self.input_mods):
            # mask = getattr(self,"src_mask_"+str(i))
            #mask = self.src_masks[i]
            latents.append(self.input_mod_nets[i].forward(data[i]))
        latent = torch.cat(latents)
        outputs = []
        for i, mod in enumerate(self.output_mods):
            # mask = getattr(self,"out_mask_"+str(i))
            #mask = self.output_masks[i]
            trans_output = self.output_mod_nets[i].forward(latent)[:self.output_lengths[i]]
            output, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
            outputs.append(output.permute(1,0,2))

        # import pdb;pdb.set_trace()
        #shape

        return outputs

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        latents = []
        for i, mod in enumerate(self.input_mods):
            # mask = getattr(self,"src_mask_"+str(i))
            latents.append(self.input_mod_nets[i].forward(self.inputs[i]))

        latent = torch.cat(latents)
        loss = 0
        for i, mod in enumerate(self.output_mods):
            # mask = getattr(self,"out_mask_"+str(i))
            output = self.output_mod_nets[i].forward(latent)[:self.output_lengths[i]]
            glow = self.output_mod_glows[i]
            # import pdb;pdb.set_trace()
            z, sldj = glow(x=self.targets[i].permute(1,0,2), cond=output.permute(1,0,2)) #time, batch, features -> batch, time, features
            loss += glow.loss_generative(z, sldj)
        self.log('nll_loss', loss)
        return loss

    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
