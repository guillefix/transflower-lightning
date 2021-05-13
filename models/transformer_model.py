import torch
from torch import nn
from .transformer import BasicTransformerModel
from models import BaseModel
from .util.generation import autoregressive_generation_multimodal

class TransformerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        opt=self.opt
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_lengths = self.input_lengths

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device, use_pos_emb=True, input_length=input_lengths[i], use_x_transformers=opt.use_x_transformers, opt=opt)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):
            net = BasicTransformerModel(douts[i], opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths), use_x_transformers=opt.use_x_transformers, opt=opt)
            # net = BasicTransformerModel(douts[i], opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=True, input_length=sum(input_lengths))
            name = "_output_"+mod
            setattr(self,"net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)

        #This is feature creep. Will remove soon
        # if self.opt.generate_attention_masks:
        self.generate_full_masks()
        self.inputs = []
        self.targets = []
        self.criterion = nn.MSELoss()

    def name(self):
        return "Transformer"

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--use_pos_emb_output', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--use_rotary_pos_emb', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--use_x_transformers', action='store_true', help="whether to use rotary position embeddings")
        # parser.add_argument('--generate_attention_masks', action='store_true', help="whether to generate the masks (but right now they are full masks, so it's not necessary")
        return parser

    def generate_full_masks(self):
        input_mods = self.input_mods
        output_mods = self.output_mods
        input_lengths = self.input_lengths
        self.src_masks = []
        for i, mod in enumerate(input_mods):
            mask = torch.zeros(input_lengths[i],input_lengths[i])
            self.register_buffer('src_mask_'+str(i), mask)
            self.src_masks.append(mask)

        self.output_masks = []
        for i, mod in enumerate(output_mods):
            mask = torch.zeros(sum(input_lengths),sum(input_lengths))
            self.register_buffer('out_mask_'+str(i), mask)
            self.output_masks.append(mask)

    def forward(self, data):
        # in lightning, forward defines the prediction/inference actions
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(data[i]))
        latent = torch.cat(latents)
        outputs = []
        for i, mod in enumerate(self.output_mods):
            output = self.output_mod_nets[i].forward(latent)[:self.output_lengths[i]]
            outputs.append(output)

        #import pdb;pdb.set_trace()
        return outputs

    def training_step(self, batch, batch_idx):
        self.set_inputs(batch)
        #print(self.inputs)
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.inputs[i]))

        latent = torch.cat(latents)
        loss_mse = 0
        for i, mod in enumerate(self.output_mods):
            output = self.output_mod_nets[i].forward(latent)[:self.output_lengths[i]]
            #print(output)
            loss_mse += self.criterion(output, self.targets[i])
            #loss_mse += self.criterion(output, self.targets[i]).detach()
        #print(loss_mse)
        #if self.opt.precision == 16:
        #    loss_mse *= 100 # loss scaling
        self.log('mse_loss', loss_mse)
        return loss_mse
        #return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    #def configure_optimizers(self):
    #    print("HIIIIIIIIIIIIIIIIII")
    #    optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.learning_rate)
    #    return [optimizer]


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
