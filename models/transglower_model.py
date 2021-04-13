import torch
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast
from torch import nn

from .util.generation import autoregressive_generation_multimodal
from .moglow.models import Glow

#TODO: refactor a whole bunch of stuff

class TransglowerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_seq_lens = self.input_seq_lens

        self.input_mod_nets = []
        self.input_mod_funcs = []
        self.output_mod_nets = []
        self.output_mod_mean_nets = []
        self.output_mod_funcs = []
        self.output_mod_glows = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device, use_pos_emb=True, input_length=input_seq_lens[i]).to(self.device)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            # self.input_mod_funcs.append(func)
            self.module_names.append(name)
        def func1(x):
            return self.input_mod_nets[0].forward(x)
        func1 = torch.vmap(func1)
        def func2(x):
            return self.input_mod_nets[1].forward(x)
        func2 = torch.vmap(func2)
        self.input_mod_funcs = [func1, func2]
        # should only be one output_mod
        for i, mod in enumerate(output_mods):
            net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_seq_lens)).to(self.device)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)
            if self.opt.residual:
                def func3(x):
                    return self.output_mod_nets[i].forward(x)
            else:
                def func3(x):
                    return self.output_mod_nets[i].forward(x)[:self.conditioning_seq_lens[i]]

            func3 = torch.vmap(func3)
            self.output_mod_funcs.append(func3)
            if opt.residual:
                net = nn.Linear(opt.dhid,douts[i])
                name="_output_mean_encoder"
                setattr(self, "net"+name, net)
                self.output_mod_mean_nets.append(net)

            cond_dim = opt.dhid
            output_dim = douts[i]
            glow = Glow(output_dim, cond_dim, self.opt)
            name = "_output_glow_"+mod
            setattr(self, "net"+name, glow)
            self.output_mod_glows.append(glow)

        self.inputs = []
        self.targets = []
        self.mean_loss = nn.MSELoss()
        self.mse_loss = 0
        self.nll_loss = 0

    def name(self):
        return "Transglower"

    def parse_base_arguments(self):
        super().parse_base_arguments()
        self.input_seq_lens = [int(x) for x in str(self.opt.input_seq_lens).split(",")]
        self.output_seq_lens = [int(x) for x in str(self.opt.output_seq_lens).split(",")]
        if self.opt.phase == "inference":
            self.input_lengths = [int(x) for x in self.opt.input_seq_lens.split(",")]
            self.output_lengths = [int(x) for x in self.opt.output_seq_lens.split(",")]
        else:
            self.input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
            self.output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]

        if self.opt.conditioning_seq_lens is not None:
            self.conditioning_seq_lens = [int(x) for x in str(self.opt.conditioning_seq_lens).split(",")]
        else:
            self.conditioning_seq_lens = [1 for x in self.opt.output_lengths.split(",")]

        if len(self.output_time_offsets) < len(self.output_mods):
            if len(self.output_time_offsets) == 1:
                self.output_time_offsets = self.output_time_offsets*len(self.output_mods)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        if len(self.input_time_offsets) < len(self.input_mods):
            if len(input_time_offsets) == 1:
                self.input_time_offsets = self.input_time_offsets*len(self.input_mods)
            else:
                raise Exception("number of input_time_offsets doesnt match number of input_mods")

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dhid_flow', type=int, default=512)
        parser.add_argument('--conditioning_seq_lens', type=str, default=None, help="the number of outputs of the conditioning transformers to feed (meaning the number of elements along the sequence dimension)")
        parser.add_argument('--input_seq_lens', type=str, default="10,11")
        parser.add_argument('--output_seq_lens', type=str, default="1")
        parser.add_argument('--glow_K', type=int, default=16)
        parser.add_argument('--actnorm_scale', type=float, default=1.0)
        parser.add_argument('--flow_permutation', type=str, default="invconv")
        parser.add_argument('--flow_coupling', type=str, default="affine")
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--network_model', type=str, default="LSTM")
        parser.add_argument('--LU_decomposed', action='store_true')
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
        parser.add_argument('--residual', action='store_true', help="whether to use the flow to predict the residual around a determnisitic mean")
        return parser

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        # min_len = min(self.input_seq_lens)
        for i,mod in enumerate(self.input_mods):
            input_ = data[i]
            input_ = input_.permute(1,2,0)
            input_ = input_.permute(0,2,1)
            input_ = self.concat_sequence(self.input_seq_lens[i], input_)
            # input_ = input_.permute(0,2,1)
            input_ = input_.permute(1,2,0,3) # L, T, B, C
            # input_ = input_[:,:,:min_len]
            # inputs_.append(input_)
            data[i] = input_

        latents = []
        for i, mod in enumerate(self.input_mods):
            # import pdb;pdb.set_trace()
            result = self.input_mod_funcs[i](data[i])
            latents.append(result)

        latent = torch.cat(latents,dim=1)
        loss = 0
        outputs = []
        if self.opt.residual:
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_funcs[i](latent).permute(2,1,3,0)
                # trans_output = []
                # for lat in latent:
                #     trans_output.append(self.output_mod_funcs[i](lat))
                # trans_output = torch.stack(trans_output).permute(2,1,3,0)
                latents = trans_output[:self.conditioning_seq_lens[i]]
                trans_predicted_mean_latents = trans_output[self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_lengths[i]]
                latents = latents.reshape(latents.shape[0], latents.shape[1] * latents.shape[2], latents.shape[3])
                predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents)
                output = glow(x=None, cond=latents, reverse=True)
                outputs.append(output.permute(0,2,1)+predicted_mean)
        else:
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_funcs[i](latent).permute(2,1,3,0)
                # trans_output = []
                # for lat in latent:
                #     trans_output.append(self.output_mod_funcs[i](lat))
                # trans_output = torch.stack(trans_output).permute(2,1,3,0)
                trans_output = trans_output.reshape(trans_output.shape[0], trans_output.shape[1] * trans_output.shape[2], trans_output.shape[3])
                output = glow(x=None, cond=trans_output, reverse=True)
                outputs.append(output.permute(0,2,1))

        return outputs

    def on_test_start(self):
        for i, mod in enumerate(self.output_mods):
            self.output_mod_glows[i].init_lstm_hidden()

    def on_train_start(self):
        for i, mod in enumerate(self.output_mods):
            self.output_mod_glows[i].init_lstm_hidden()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        for i, mod in enumerate(self.output_mods):
            self.output_mod_glows[i].init_lstm_hidden()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        for i, mod in enumerate(self.output_mods):
            self.output_mod_glows[i].init_lstm_hidden()

    def concat_sequence(self, seqlen, data):
        #NOTE: this could be done as preprocessing on the dataset to make it a bit more efficient, but we are only going to
        # use this for baseline moglow, so I thought it wasn't worth it to put it there.
        """
        Concatenates a sequence of features to one.
        """
        nn,n_timesteps,n_feats = data.shape
        L = n_timesteps-(seqlen-1)
        # import pdb;pdb.set_trace()
        inds = torch.zeros((L, seqlen), dtype=torch.long)

        #create indices for the sequences we want
        rng = torch.arange(0, n_timesteps, dtype=torch.long)
        for ii in range(0,seqlen):
            # print(rng[ii:(n_timesteps-(seqlen-ii-1))].shape)
            # inds[:, ii] = torch.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))], 0, 1)
            inds[:, ii] = rng[ii:(n_timesteps-(seqlen-ii-1))]

        #slice each sample into L sequences and store as new samples
        cc=data[:,inds,:].clone()

        #print ("cc: " + str(cc.shape))

        #reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen, n_feats))
        #print ("dd: " + str(dd.shape))
        return dd

    def set_inputs(self, data):
        self.inputs = []
        self.targets = []
        for i, mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_shape = input_.shape
            if self.input_seq_lens[i] > 1:
                # input_ = input_.permute(0,2,1)
                input_ = self.concat_sequence(self.input_seq_lens[i], input_)
                # input_ = input_.permute(0,2,1)
            else:
                input_ = input_.permute(0,2,1)
                input_ = input_.squeeze(2)
            input_ = input_.permute(1,2,0,3) # L, T, B, C
            self.inputs.append(input_)
        for i, mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_shape = target_.shape
            if self.output_seq_lens[i] > 1:
                # target_ = target_.permute(0,2,1)
                target_ = self.concat_sequence(self.output_seq_lens[i], target_)
                target_ = target_.permute(0,2,1)
            else:
                target_ = target_.permute(0,2,1)
            self.targets.append(target_)

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        latents = []
        for i, mod in enumerate(self.input_mods):
            # import pdb;pdb.set_trace()
            result = self.input_mod_funcs[i](self.inputs[i])
            # result = []
            # for inp in self.inputs[i]:
            #     result.append(self.input_mod_funcs[i](inp))
            # result = torch.stack(result)
            latents.append(result)

        latent = torch.cat(latents,dim=1)
        if self.opt.residual:
            nll_loss = 0
            mse_loss = 0
            for i, mod in enumerate(self.output_mods):
                trans_output = self.output_mod_funcs[i](latent)
                # trans_output = []
                # for lat in latent:
                #     trans_output.append(self.output_mod_funcs[i](lat))
                # trans_output = torch.stack(trans_output)
                latents = trans_output[:,:self.conditioning_seq_lens[i]].permute(2,1,3,0)
                trans_predicted_mean_latents = trans_output[:,self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_seq_lens[i]]
                latents = latents.reshape(latents.shape[0], latents.shape[1] * latents.shape[2], latents.shape[3])
                trans_predicted_mean_latents = trans_predicted_mean_latents.reshape(trans_predicted_mean_latents.shape[0], trans_predicted_mean_latents.shape[1] * trans_predicted_mean_latents.shape[2], trans_predicted_mean_latents.shape[3])
                predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents).permute(1,2,0)
                glow = self.output_mod_glows[i]
                # import pdb;pdb.set_trace()
                z, nll = glow(x=self.targets[i]-predicted_mean, cond=latents) #time, batch, features -> batch, time, features
                nll_loss += Glow.loss_generative(nll)
                mse_loss += 100*self.mean_loss(predicted_mean, self.targets[i])
            loss = nll_loss + mse_loss
            self.mse_loss = mse_loss
            self.nll_loss = nll_loss
            self.log('mse_loss', mse_loss)
            self.log('nll_loss', nll_loss)
        else:
            loss = 0
            for i, mod in enumerate(self.output_mods):
                output = self.output_mod_funcs[i](latent).permute(2,1,3,0)
                # output = []
                # for lat in latent:
                #     output.append(self.output_mod_funcs[i](lat))
                # output = torch.stack(output).permute(2,1,3,0)
                output = output.reshape(output.shape[0], output.shape[1] * output.shape[2], output.shape[3])
                glow = self.output_mod_glows[i]
                # import pdb;pdb.set_trace()
                z, nll = glow(x=self.targets[i], cond=output) #time, batch, features -> batch, time, features
                loss += Glow.loss_generative(nll)
        self.log('loss', loss)
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
