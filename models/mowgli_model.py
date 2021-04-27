import torch
import numpy as np
from models.transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast
from torch import nn

from .util.generation import autoregressive_generation_multimodal

from models.cdvae import ConditionalDiscreteVAE

class MowgliModel(BaseModel):
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
        self.output_mod_mean_nets = []
        self.output_mod_vaes = []
        self.conditioning_seq_lens = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device, use_pos_emb=True, input_length=input_lengths[i], use_x_transformers=opt.use_x_transformers, opt=opt)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):

            # import pdb;pdb.set_trace()
            vae = ConditionalDiscreteVAE(
                input_shape = (output_lengths[i], 1),
                channels = douts[i],
                num_layers = opt.vae_num_layers,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
                num_tokens = opt.vae_num_tokens,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
                codebook_dim = opt.vae_codebook_dim,       # codebook dimension
                hidden_dim = opt.vae_dhid,          # hidden dimension
                num_resnet_blocks = opt.vae_num_resnet_blocks,    # number of resnet blocks
                temperature = opt.vae_temp,        # gumbel softmax temperature, the lower this is, the harder the discretization
                straight_through = opt.vae_hard, # straight-through for gumbel softmax. unclear if it is better one way or the other
                cond_dim = opt.dhid,
                prior_nhead = opt.prior_nhead,
                prior_dhid = opt.prior_dhid,
                prior_nlayers = opt.prior_nlayers,
                prior_dropout = opt.prior_dropout,
                prior_use_pos_emb = not opt.prior_no_use_pos_emb,
                prior_use_x_transformers = opt.prior_use_x_transformers,
                opt = opt
            )

            name = "_output_vae_"+mod
            setattr(self, "net"+name, vae)
            self.output_mod_vaes.append(vae)

            self.conditioning_seq_lens.append(np.prod(vae.codebook_layer_shape))
            net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths), use_x_transformers=opt.use_x_transformers, opt=opt)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)
            if opt.residual:
                if self.opt.cond_concat_dims:
                    net = nn.Linear(opt.dhid,douts[i])
                else:
                    net = nn.Linear(opt.dhid,opt.douts[i])
                name="_output_mean_encoder"
                setattr(self, "net"+name, net)
                self.output_mod_mean_nets.append(net)


        self.mean_loss = nn.MSELoss()
        self.inputs = []
        self.targets = []
        self.mse_loss = 0
        self.nll_loss = 0
        self.prior_loss_weight = opt.prior_loss_weight_initial

    def name(self):
        return "mowgli"

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        # parser.add_argument('--conditioning_seq_lens', type=str, default=None, help="the number of outputs of the conditioning transformers to feed (meaning the number of elements along the sequence dimension)")
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--vae_num_layers', type=int, default=3)
        parser.add_argument('--vae_num_tokens', type=int, default=2048)
        parser.add_argument('--vae_codebook_dim', type=int, default=512)
        parser.add_argument('--vae_dhid', type=int, default=64)
        parser.add_argument('--prior_dhid', type=int, default=512)
        parser.add_argument('--prior_nhead', type=int, default=8)
        parser.add_argument('--prior_nlayers', type=int, default=8)
        parser.add_argument('--prior_dropout', type=float, default=0)
        parser.add_argument('--prior_loss_weight_initial', type=float, default=0)
        parser.add_argument('--prior_loss_weight_warmup_epochs', type=float, default=500)
        parser.add_argument('--max_prior_loss_weight', type=float, default=0, help="max value of prior loss weight during stage 1 (e.g. 0.01 is a good value)")
        parser.add_argument('--vae_num_resnet_blocks', type=int, default=1)
        parser.add_argument('--vae_temp', type=float, default=0.9)
        parser.add_argument('--vae_hard', action='store_true', help="whether to use the hard one-hot vector as output and use the straight through gradient estimator, for discrete latents")
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--scales', type=str, default="[[10,0]]")
        parser.add_argument('--residual', action='store_true', help="whether to use the vae to predict the residual around a determnisitic mean")
        parser.add_argument('--use_pos_emb_output', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--use_rotary_pos_emb', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--use_x_transformers', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--prior_use_x_transformers', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--prior_no_use_pos_emb', action='store_true', help="dont use positional embeddings for the prior transformer")
        parser.add_argument('--stage2', action='store_true', help="stage2: train the prior, rather than the VAE")
        return parser

    def forward(self, data, temp=1.0):
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
                # residual, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
                residual = self.output_mod_vaes[i].generate(trans_output.permute(1,2,0), temp=temp)
                residual = residual.squeeze(-1)
                output = predicted_mean + residual.permute(2,0,1)
                outputs.append(output)
        else:
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
                output = self.output_mod_vaes[i].generate(trans_output.permute(1,2,0), temp=temp)
                # import pdb;pdb.set_trace()
                output = output.squeeze(-1)
                outputs.append(output.permute(2,0,1))
        return outputs

    def on_train_epoch_start(self):
        self.prior_loss_weight = self.opt.max_prior_loss_weight * min((self.opt.prior_loss_weight_warmup_epochs - self.current_epoch)/self.opt.prior_loss_weight_warmup_epochs, 1)

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
            accuracies = []
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_nets[i].forward(latent)
                latents = trans_output[:self.conditioning_seq_lens[i]]
                trans_predicted_mean_latents = trans_output[self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_lengths[i]]
                predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents)
                vae = self.output_mod_vaes[i]
                if not self.opt.stage2:
                    nll_loss += vae((self.targets[i] - predicted_mean).permute(1,2,0), cond=latents.permute(1,2,0), return_loss=True) #time, batch, features -> batch, features, time
                    if self.opt.max_prior_loss_weight > 0:
                        prior_loss, accuracy = vae.prior_logp((self.targets[i] - predicted_mean).permute(1,2,0), cond=latents.permute(1,2,0), return_accuracy=True)
                        accuracies.append(accuracy)
                        nll_loss += self.prior_loss_weight * prior_loss
                else:
                    prior_loss, accuracy = vae.prior_logp((self.targets[i] - predicted_mean).permute(1,2,0), cond=latents.permute(1,2,0), return_accuracy=True, detach_cond=True)
                    nll_loss += prior_loss
                    accuracies.append(accuracy)
                mse_loss += 100*self.mean_loss(predicted_mean[i], self.targets[i])
            loss = nll_loss + mse_loss
            self.mse_loss = mse_loss
            self.nll_loss = nll_loss
            self.log('mse_loss', mse_loss)
            self.log('nll_loss', nll_loss)
            if len(accuracies) > 0:
                self.log('accuracy', torch.mean(torch.stack(accuracies)))
        else:
            loss = 0
            accuracies = []
            # import pdb;pdb.set_trace()
            for i, mod in enumerate(self.output_mods):
                output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
                vae = self.output_mod_vaes[i]
                if not self.opt.stage2:
                    loss += vae(self.targets[i].permute(1,2,0), cond=output.permute(1,2,0), return_loss=True) #time, batch, features -> batch, features, time
                    if self.opt.max_prior_loss_weight > 0:
                        prior_loss, accuracy = vae.prior_logp(self.targets[i].permute(1,2,0), cond=output.permute(1,2,0), return_accuracy=True)
                        loss += self.prior_loss_weight * prior_loss
                        accuracies.append(accuracy)
                else:
                    prior_loss, accuracy = vae.prior_logp(self.targets[i].permute(1,2,0), cond=output.permute(1,2,0), return_accuracy=True, detach_cond=True)
                    ##prior_loss, accuracy = vae.prior_logp(self.targets[i].permute(1,2,0), return_accuracy=True, detach_cond=True)
                    loss += prior_loss
                    accuracies.append(accuracy)

        self.log('loss', loss)
        if len(accuracies) > 0:
           self.log('accuracy', torch.mean(torch.stack(accuracies)))
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
