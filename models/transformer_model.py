import torch
from torch import nn
from .transformer import BasicTransformerModel
from models import BaseModel
from .util.generation import autoregressive_generation_multimodal

class TransformerModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        self.douts = douts = [int(x) for x in self.opt.douts.split(",")]
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        self.predicted_inputs = predicted_inputs = [int(x) for x in self.opt.predicted_inputs.split(",")]
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

        if len(predicted_inputs) < len(input_mods):
            if len(predicted_inputs) == 1:
                self.predicted_inputs = predicted_inputs = predicted_inputs*len(input_mods)
            else:
                raise Exception("number of predicted_inputs doesnt match number of input_mods")

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device, use_pos_emb=True, input_length=input_lengths[i])
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):
            net = BasicTransformerModel(douts[i], opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths))
            name = "_output_"+mod
            setattr(self,"net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)

        #This is feature creep. Will remove soon
        if self.opt.generate_attention_masks:
            self.generate_full_masks()
        self.inputs = []
        self.targets = []
        self.criterion = nn.MSELoss()

    def name(self):
        return "Transformer"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dins', default=None)
        parser.add_argument('--douts', default=None)
        parser.add_argument('--predicted_inputs', default="0")
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--use_pos_emb_output', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--generate_attention_masks', action='store_true', help="whether to generate the masks (but right now they are full masks, so it's not necessary")
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

        # import pdb;pdb.set_trace()
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
        self.log('mse_loss', loss_mse)
        return loss_mse
        #return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    def test_step(self, batch, batch_idx):
        self.set_inputs(batch)
        #print(self.inputs)
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
        # print(loss_mse)
        return loss_mse
        # return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    #def configure_optimizers(self):
    #    print("HIIIIIIIIIIIIIIIIII")
    #    optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.learning_rate)
    #    return [optimizer]


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
