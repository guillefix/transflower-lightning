from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from models.transformer import BasicTransformerModel, EncDecTransformerModel, EncDecXTransformer

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

# from dalle_pytorch import distributed_utils
# from dalle_pytorch.vae import OpenAIDiscreteVAE
# from dalle_pytorch.vae import VQGanVAE1024
# from dalle_pytorch.transformer import Transformer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class ConditionalDiscreteVAEVision(nn.Module):
    def __init__(
            self,
            image_shape = (256,256),
            num_tokens = 512,
            codebook_dim = 512,
            num_layers = 3,
            num_resnet_blocks = 0,
            hidden_dim = 64,
            conditioning_dim = 64,
            channels = 3,
            smooth_l1_loss = False,
            temperature = 0.9,
            straight_through = False,
            kl_div_loss_weight = 0.,
            normalization = ((0.5,) * 3, (0.5,) * 3)
    ):
        super().__init__()
        assert log2(image_shape[0]).is_integer(), 'image size must be a power of 2'
        assert log2(image_shape[1]).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_shape = image_shape
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        if not has_resblocks:
            dec_init_chan = codebook_dim
        else:
            dec_init_chan = dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

        # self._register_external_parameters()

    # def _register_external_parameters(self):
    #     """Register external parameters for DeepSpeed partitioning."""
    #     if (
    #             not distributed_utils.is_distributed
    #             or not distributed_utils.using_backend(
    #         distributed_utils.DeepSpeedBackend)
    #     ):
    #         return
    #
    #     deepspeed = distributed_utils.backend.backend_module
    #     deepspeed.zero.register_external_parameters(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
            self,
            img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
            self,
            img,
            return_loss = False,
            return_recons = False,
            return_logits = False,
            temp = None
    ):
        device, num_tokens, image_shape, kl_div_loss_weight = img.device, self.num_tokens, self.image_shape, self.kl_div_loss_weight
        assert img.shape[-1] == image_shape[1] and img.shape[-2] == image_shape[0], f'input must have the correct image size {image_shape[0]}x{image_shape[1]}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out

class ConditionalDiscreteVAE(nn.Module):
    def __init__(
            self,
            input_shape = (256,256),
            num_tokens = 512,
            codebook_dim = 512,
            num_layers = 3,
            num_resnet_blocks = 0,
            hidden_dim = 64,
            cond_dim = 0,
            channels = 3,
            smooth_l1_loss = False,
            temperature = 0.9,
            straight_through = False,
            kl_div_loss_weight = 0.,
            normalization = None,
            prior_nhead = 8,
            prior_dhid = 512,
            prior_nlayers = 8,
            prior_dropout = 0,
            prior_use_pos_emb = True,
            prior_use_x_transformers = False,
            opt = None,
            cond_vae = False
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.input_shape = input_shape
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)
        self.cond_dim = cond_dim
        self.cond_vae = cond_vae

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        if cond_vae:
            enc_chans = [channels + cond_dim, *enc_chans]
        else:
            enc_chans = [channels, *enc_chans]

        if not has_resblocks:
            if cond_vae:
                dec_init_chan = codebook_dim + cond_dim
            else:
                dec_init_chan = codebook_dim
        else:
            dec_init_chan = dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []


        if input_shape[0] == 1:
            kernel_size1 = 1
            padding_size1 = 0
            codebook_layer_shape1 = 1
        elif input_shape[0] in [2,3,4]:
            kernel_size1 = 3
            padding_size1 = 1
            codebook_layer_shape1 = input_shape[0]
        else:
            #kernel_size1 = 4
            kernel_size1 = 3
            padding_size1 = 1
            #codebook_layer_shape1 = input_shape[0] - num_layers
            codebook_layer_shape1 = input_shape[0]

        if input_shape[1] == 1:
            kernel_size2 = 1
            padding_size2 = 0
            codebook_layer_shape2 = 1
        elif input_shape[1] in [2,3,4]:
            kernel_size2 = 3
            padding_size2 = 1
            codebook_layer_shape2 = input_shape[1]
        else:
            #kernel_size2 = 4
            kernel_size2 = 3
            padding_size2 = 1
            #codebook_layer_shape2 = input_shape[1] - num_layers
            codebook_layer_shape2 = input_shape[1]

        self.codebook_layer_shape = (codebook_layer_shape1,codebook_layer_shape2)
        kernel_shape = (kernel_size1, kernel_size2)
        padding_shape = (padding_size1, padding_size2)
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, kernel_shape, stride = 1, padding = padding_shape), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, kernel_shape, stride = 1, padding = padding_shape), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            if cond_vae:
                dec_layers.insert(0, nn.Conv2d(codebook_dim + cond_dim, dec_chans[1], 1))
            else:
                dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.cond_upsampler = torch.nn.Upsample(size=input_shape) #upsampler to feed the conditioning to the input of the encoder
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

        latent_size = codebook_layer_shape1*codebook_layer_shape2
        self.latent_size = latent_size
        if cond_dim > 0:
            self.prior_transformer = ContDiscTransformer(cond_dim, num_tokens, codebook_dim, prior_nhead, prior_dhid, prior_nlayers, prior_dropout,
                                                         use_pos_emb=prior_use_pos_emb,
                                                         src_length=latent_size,
                                                         tgt_length=latent_size,
                                                         use_x_transformers=prior_use_x_transformers,
                                                         opt=opt)

        # self._register_external_parameters()

    # def _register_external_parameters(self):
    #     """Register external parameters for DeepSpeed partitioning."""
    #     if (
    #             not distributed_utils.is_distributed
    #             or not distributed_utils.using_backend(
    #         distributed_utils.DeepSpeedBackend)
    #     ):
    #         return
    #
    #     deepspeed = distributed_utils.backend.backend_module
    #     deepspeed.zero.register_external_parameters(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, inputs, cond=None):
        logits = self(inputs, cond, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
            self,
            img_seq,
            cond = None
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        if cond is not None:
            image_embeds_cond = torch.cat([image_embeds, cond], dim = 1)
            images = self.decoder(image_embeds_cond)
        else:
            images = self.decoder(image_embeds)

        return images

    def prior_logp(
            self,
            inputs,
            cond = None,
            return_accuracy = False,
            detach_cond = False
       ):
        # import pdb;pdb.set_trace()
        #if cond is None: raise NotImplementedError("Haven't implemented non-conditional DVAEs")
        if len(inputs.shape) == 3:
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1],*self.input_shape)
        if len(cond.shape) == 3:
            cond = cond.reshape(cond.shape[0], cond.shape[1],*self.codebook_layer_shape)
        with torch.no_grad():
            if self.cond_vae:
                labels = self.get_codebook_indices(inputs, cond)
            else:
                labels = self.get_codebook_indices(inputs)
        if detach_cond:
            cond = cond.detach()
        logits = self.prior_transformer(cond.squeeze(-1).permute(2,0,1), labels.permute(1,0)).permute(1,2,0)
        loss = F.cross_entropy(logits, labels)
        if not return_accuracy:
            return loss
        # import pdb;pdb.set_trace()
        predicted = logits.argmax(dim = 1).flatten(1)
        accuracy = (predicted == labels).sum()/predicted.nelement()
        return loss, accuracy

    def generate(self, cond, temp=1.0, filter_thres = 0.5):
        #if cond is None: raise NotImplementedError("Haven't implemented non-conditional DVAEs")
        if len(cond.shape) == 3:
            cond = cond.reshape(cond.shape[0], cond.shape[1],*self.codebook_layer_shape)
        dummy = torch.zeros(1,1).long().to(cond.device)
        tokens = []
        for i in range(self.latent_size):
            # print(i)
            logits = self.prior_transformer(cond.squeeze(-1).permute(2,0,1), torch.cat(tokens+[dummy], 0)).permute(1,2,0)[:,-1,:]
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temp, dim = -1)
            sampled = torch.multinomial(probs, 1)
            tokens.append(sampled)
        print(tokens)
        embs = self.codebook(torch.cat(tokens, 0))
        # import pdb;pdb.set_trace()
        if self.cond_vae:
            sampled_cond = torch.cat([embs.permute(2,0,1).unsqueeze(0),cond], dim=1)
        else:
            sampled_cond = embs.permute(2,0,1).unsqueeze(0)
        out = self.decoder(sampled_cond)
        return out

    def forward(
            self,
            inp,
            cond = None,
            return_loss = False,
            return_recons = False,
            return_logits = False,
            temp = None
    ):
        if len(inp.shape) == 3:
            inp = inp.reshape(inp.shape[0], inp.shape[1],*self.input_shape)
        device, num_tokens, input_shape, kl_div_loss_weight = inp.device, self.num_tokens, self.input_shape, self.kl_div_loss_weight
        assert inp.shape[-1] == input_shape[1] and inp.shape[-2] == input_shape[0], f'input must have the correct image size {input_shape[0]}x{input_shape[1]}. Instead got {inp.shape[0]}x{inp.shape[1]}'

        inp = self.norm(inp)
        if cond is not None:
            if len(cond.shape) == 3:
                cond = cond.reshape(cond.shape[0], cond.shape[1],*self.codebook_layer_shape)
            cond_upsampled = self.cond_upsampler(cond)
            inp_cond = torch.cat([inp,cond_upsampled], dim=1)
            inp_cond = self.norm(inp_cond)
        else:
            inp_cond = self.norm(inp)

        logits = self.encoder(inp_cond)
        # codebook_indices = logits.argmax(dim = 1).flatten(1)
        # print(codebook_indices.shape)
        # print(codebook_indices)
        # print(list(self.encoder.parameters())[1].data)
        # for p in self.prior_transformer.parameters():
        #     print(p.norm())

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        if cond is not None:
            sampled_cond = torch.cat([sampled,cond], dim=1)
            out = self.decoder(sampled_cond)
        else:
            out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        # import pdb;pdb.set_trace()
        recon_loss = self.loss_fn(inp, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out

class ContDiscTransformer(nn.Module):

    def __init__(self, src_d, tgt_num_tokens, tgt_emb_dim, nhead, dhid, nlayers, dropout=0.5,use_pos_emb=False,src_length=0,tgt_length=0,use_x_transformers=False,opt=None):
        super(ContDiscTransformer, self).__init__()
        self.transformer = EncDecTransformerModel(tgt_num_tokens, src_d, tgt_emb_dim, nhead, dhid, nlayers, dropout=dropout,use_pos_emb=use_pos_emb,src_length=src_length,tgt_length=tgt_length,use_x_transformers=use_x_transformers,opt=opt)
        #self.transformer = EncDecTransformerModel(tgt_num_tokens, src_d, tgt_emb_dim, nhead, dhid, nlayers, dropout=dropout,use_pos_emb=False,src_length=src_length,tgt_length=tgt_length,use_x_transformers=use_x_transformers,opt=opt)
        # self.transformer = EncDecXTransformer(dim=dhid, dec_dim_out=tgt_num_tokens, enc_dim_in=src_d, enc_dim_out=tgt_emb_dim, dec_din_in=tgt_emb_dim, enc_heads=nhead, dec_heads=nhead, enc_depth=nlayers, dec_depth=nlayers, enc_dropout=dropout, dec_dropout=dropout, enc_max_seq_len=1024, dec_max_seq_len=1024)
        self.embedding = nn.Embedding(tgt_num_tokens, tgt_emb_dim)
        self.first_input = nn.Parameter((torch.randn(1,1,tgt_emb_dim)))

    def forward(self, src, tgt):
        tgt = tgt[:-1]
        embs = self.embedding(tgt)
        embs = torch.cat([torch.tile(self.first_input, (1,embs.shape[1],1)), embs], 0)
        output = self.transformer(src,embs)
        return output
