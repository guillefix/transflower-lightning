import torch
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast
from torch import nn
import torch.nn.functional as F

from .util.generation import autoregressive_generation_multimodal

class TransflowerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        if self.opt.conditioning_seq_lens is not None:
            self.conditioning_seq_lens = [int(x) for x in str(self.opt.conditioning_seq_lens).split(",")]
        else:
            self.conditioning_seq_lens = [int(x) for x in str(self.opt.output_lengths).split(",")]

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.output_mod_mean_nets = []
        self.output_mod_glows = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout,
                                        ntokens=self.input_num_tokens[i],
                                        use_pos_emb=opt.use_pos_emb_inputs,
                                        use_rel_pos_emb=opt.use_rel_pos_emb_inputs,
                                        input_length=input_lengths[i],
                                        use_x_transformers=opt.use_x_transformers,
                                        opt=opt,
                                        discrete_inputs=self.input_types[i] == 'd')
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):
            # if self.opt.cond_concat_dims:
            net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout,
                                        ntokens=self.output_num_tokens[i], # tho not being used yet
                                        use_pos_emb=opt.use_pos_emb_output,
                                        use_rel_pos_emb=opt.use_rel_pos_emb_output,
                                        input_length=sum(input_lengths),
                                        use_x_transformers=opt.use_x_transformers,
                                        opt=opt)
            # else:
            #     net = BasicTransformerModel(douts[i]//2, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths), use_x_transformers=opt.use_x_transformers, opt=opt)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)
            if opt.residual:
                # if self.opt.cond_concat_dims:
                net = nn.Linear(opt.dhid,douts[i])
                # else:
                #     net = nn.Linear(douts[i]//2,douts[i])
                name="_output_mean_encoder"
                setattr(self, "net"+name, net)
                self.output_mod_mean_nets.append(net)

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
                                     use_rel_pos_emb=opt.use_rel_pos_emb_coupling,
                                     norm_layer = opt.glow_norm_layer,
                                     bn_momentum = opt.glow_bn_momentum,
                                     cond_concat_dims=opt.cond_concat_dims,
                                     flow_dist=opt.flow_dist,
                                     flow_dist_param=opt.flow_dist_param,
                                     cond_seq_len=self.conditioning_seq_lens[i],
                                )
            name = "_output_glow_"+mod
            setattr(self, "net"+name, glow)
            self.output_mod_glows.append(glow)

        self.mean_loss = nn.MSELoss()
        #This is feature creep. Will remove soon
        # self.generate_full_masks()
        self.inputs = []
        self.targets = []
        self.mse_loss = 0
        self.nll_loss = 0

    def name(self):
        return "Transflower"

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dhid_flow', type=int, default=512)
        parser.add_argument('--conditioning_seq_lens', type=str, default=None, help="the number of outputs of the conditioning transformers to feed (meaning the number of elements along the sequence dimension)")
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--num_heads_flow', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--scales', type=str, default="[[10,0]]")
        parser.add_argument('--flow_dist', type=str, default="normal")
        parser.add_argument('--flow_dist_param', type=int, default=50)
        parser.add_argument('--glow_norm_layer', type=str, default=None)
        parser.add_argument('--glow_bn_momentum', type=float, default=0.1)
        parser.add_argument('--num_glow_coupling_blocks', type=int, default=10)
        parser.add_argument('--num_mixture_components', type=int, default=0)
        parser.add_argument('--glow_use_attn', action='store_true', help="whether to use the internal attention for the FlowPlusPLus model")
        parser.add_argument('--use_transformer_nn', action='store_true', help="whether to use the internal attention for the FlowPlusPLus model")
        parser.add_argument('--use_rel_pos_emb_inputs', action='store_true', help="whether to use T5 relative positional embeddings for input modality transformers")
        parser.add_argument('--use_rel_pos_emb_output', action='store_true', help="whether to use T5 relative positional embeddings for output modality transformers")
        parser.add_argument('--use_pos_emb_inputs', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--use_pos_emb_output', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--use_pos_emb_coupling', action='store_true', help="whether to use positional embeddings for the coupling layer transformers")
        parser.add_argument('--use_rel_pos_emb_coupling', action='store_true', help="whether to use T5 relative positional embeddings for the coupling layer transformers")
        parser.add_argument('--cond_concat_dims', action='store_true', help="if set we concatenate along the channel dimension with with the x for the coupling layer; otherwise we concatenate along the sequence dimesion")
        parser.add_argument('--residual', action='store_true', help="whether to use the flow to predict the residual around a determnisitic mean")
        parser.add_argument('--use_rotary_pos_emb', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--use_x_transformers', action='store_true', help="whether to use rotary position embeddings")
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
            latents.append(self.input_mod_nets[i].forward(data[i]))
        latent = torch.cat(latents)
        outputs = []
        if self.opt.residual:
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
                trans_predicted_mean_latents = self.output_mod_nets[i].forward(latent)[self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_lengths[i]]
                predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents)
                residual, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
                output = predicted_mean + residual.permute(1,0,2)
                outputs.append(output)
        else:
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
                output, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
                outputs.append(output.permute(1,0,2))

        return outputs

    # def on_train_epoch_start(self):
    #     if self.opt.residual:
    #         self.residual_loss_weight = self.opt.max_residual_loss_weight * min((self.opt.residual_loss_weight_warmup_epochs - self.current_epoch)/self.opt.residual_loss_weight_warmup_epochs, 1)

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        # print(self.input_mod_nets[0].encoder1.weight.data)
        # print(self.targets[0])
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.inputs[i]))

        latent = torch.cat(latents)
        # print(latent)
        if self.opt.residual:
            nll_loss = 0
            mse_loss = 0
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_nets[i].forward(latent)
                latents = trans_output[:self.conditioning_seq_lens[i]]
                trans_predicted_mean_latents = trans_output[self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_lengths[i]]
                predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents)
                glow = self.output_mod_glows[i]
                z, sldj = glow(x=self.targets[i].permute(1,0,2) - predicted_mean.permute(1,0,2), cond=latents.permute(1,0,2)) #time, batch, features -> batch, time, features
                nll_loss += glow.loss_generative(z, sldj)
                # import pdb;pdb.set_trace()
                mse_loss += 100*self.mean_loss(predicted_mean, self.targets[i])

            adaptive_weight = F.sigmoid(mse_loss/5)
            loss = (1-adaptive_weight)*nll_loss + adaptive_weight*mse_loss
            self.mse_loss = mse_loss
            self.nll_loss = nll_loss
            #print(mse_loss)
            #print(nll_loss)
            self.log('mse_loss', mse_loss)
            self.log('nll_loss', nll_loss)
        else:
            loss = 0
            for i, mod in enumerate(self.output_mods):
                output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
                glow = self.output_mod_glows[i]
                # import pdb;pdb.set_trace()
                # print(output)
                z, sldj = glow(x=self.targets[i].permute(1,0,2), cond=output.permute(1,0,2)) #time, batch, features -> batch, time, features
                # print(z)
                #print(sldj)
                # n_timesteps = self.targets[i].shape[1]
                loss += glow.loss_generative(z, sldj)

        self.log('loss', loss)
        # print(loss)
        return loss

    def test_step(self, batch, batch_idx):
        if self.opt.residual:
            self.eval()
            loss = self.training_step(batch, batch_idx)
            # print(loss)
            return {"test_loss": loss, "test_mse_loss": self.mse_loss, "test_nll_loss": self.nll_loss}
        else:
            return super().test_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        if self.opt.residual:
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            avg_mse_loss = torch.stack([x['test_mse_loss'] for x in outputs]).mean()
            avg_nll_loss = torch.stack([x['test_nll_loss'] for x in outputs]).mean()
            logs = {'test_loss': avg_loss, 'test_mse_loss': avg_mse_loss, 'test_nll_loss': avg_nll_loss}

            return {'log': logs}
        else:
            return super().test_epoch_end(outputs)

    #to help debug XLA stuff, like missing ops, or data loading/compiling bottlenecks
    # see https://youtu.be/iwtpwQRdb3Y?t=1056
    # def on_epoch_end(self):
    #    import torch_xla.core.xla_model as xm
    #    import torch_xla.debug.metrics as met
    #    xm.master_print(met.metrics_report())


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
