import torch
from .transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast

from .util.generation import autoregressive_generation_multimodal
from .moglow.models import Glow

#TODO: refactor a whole bunch of stuff

class TransglowerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        self.douts = douts = [int(x) for x in self.opt.douts.split(",")]
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        self.input_seq_lens = input_seq_lens = [int(x) for x in self.opt.input_seq_lens.split(",")]
        self.output_seq_lens = output_seq_lens = [int(x) for x in self.opt.output_seq_lens.split(",")]
        self.predicted_inputs = predicted_inputs = [int(x) for x in self.opt.predicted_inputs.split(",")]
        self.output_lengths = output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]
        if self.opt.conditioning_seq_lens is not None:
            self.conditioning_seq_lens = [int(x) for x in self.opt.conditioning_seq_lens.split(",")]
        else:
            self.conditioning_seq_lens = [1 for x in self.opt.output_lengths.split(",")]
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

        if len(predicted_inputs) < len(input_mods):
            if len(predicted_inputs) == 1:
                self.predicted_inputs = predicted_inputs = predicted_inputs*len(input_mods)
            else:
                raise Exception("number of predicted_inputs doesnt match number of input_mods")

        self.input_mod_nets = []
        self.input_mod_funcs = []
        self.output_mod_nets = []
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
        for i, mod in enumerate(output_mods):
            net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_seq_lens)).to(self.device)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)
            def func3(x):
                return self.output_mod_nets[i].forward(x)[:self.conditioning_seq_lens[i]]
            func3 = torch.vmap(func3)
            self.output_mod_funcs.append(func3)

            cond_dim = opt.dhid
            output_dim = douts[i]
            glow = Glow(output_dim, cond_dim, self.opt)
            name = "_output_glow_"+mod
            setattr(self, "net"+name, glow)
            self.output_mod_glows.append(glow)

        self.inputs = []
        self.targets = []

    def name(self):
        return "Transflower"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dhid_flow', type=int, default=512)
        parser.add_argument('--dins', default=None)
        parser.add_argument('--douts', default=None)
        parser.add_argument('--predicted_inputs', default="0")
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
        for i, mod in enumerate(self.output_mods):
            trans_output = self.output_mod_funcs[i](latent).permute(2,1,3,0)
            trans_output = trans_output.reshape(trans_output.shape[0], trans_output.shape[1] * trans_output.shape[2], trans_output.shape[3])
            glow = self.output_mod_glows[i]
            # import pdb;pdb.set_trace()
            output = glow(x=None, cond=trans_output, reverse=True)
            # outputs.append(output.permute(1,0,2))
            outputs.append(output.permute(0,2,1))

        return outputs

    def on_train_start(self):
        for i, mod in enumerate(self.output_mods):
            self.output_mod_glows[i].init_lstm_hidden()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # self.zero_grad()
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

    def set_inputs(self, data):
        self.inputs = []
        self.targets = []
        for i, mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_shape = input_.shape
            if len(input_shape)==4:
                # It's coming as 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
                input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3]))
            if self.input_seq_lens[i] > 1:
                input_ = input_.permute(0,2,1)
                input_ = self.concat_sequence(self.input_seq_lens[i], input_)
                # input_ = input_.permute(0,2,1)
            else:
                input_ = input_.squeeze(2)
            input_ = input_.permute(1,2,0,3) # L, T, B, C
            self.inputs.append(input_)
        for i, mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_shape = target_.shape
            if len(target_shape)==4:
                target_ = target_.reshape((target_shape[0]*target_shape[1], target_shape[2], target_shape[3]))
            if self.output_seq_lens[i] > 1:
                target_ = target_.permute(0,2,1)
                target_ = self.concat_sequence(self.output_seq_lens[i], target_)
                target_ = target_.permute(0,2,1)
            # target_ = target_.permute(2,0,1)
            self.targets.append(target_)

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        latents = []
        for i, mod in enumerate(self.input_mods):
            # import pdb;pdb.set_trace()
            result = self.input_mod_funcs[i](self.inputs[i])
            latents.append(result)

        latent = torch.cat(latents,dim=1)
        loss = 0
        for i, mod in enumerate(self.output_mods):
            output = self.output_mod_funcs[i](latent).permute(2,1,3,0)
            output = output.reshape(output.shape[0], output.shape[1] * output.shape[2], output.shape[3])
            glow = self.output_mod_glows[i]
            # import pdb;pdb.set_trace()
            z, nll = glow(x=self.targets[i], cond=output) #time, batch, features -> batch, time, features
            #print(sldj)
            n_timesteps = self.targets[i].shape[1]
            # loss += glow.loss_generative(z, sldj)
            # loss += glow.loss_generative(z, sldj)
            loss += Glow.loss_generative(nll)
        #self.log('nll_loss', loss)
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
