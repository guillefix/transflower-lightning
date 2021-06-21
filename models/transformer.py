import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
sys.path.append(THIS_DIR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import numpy as np

from functools import partial

from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
#from models.x_transformers import ContinuousTransformerWrapper, Decoder, Encoder, AutoregressiveWrapper
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder, AutoregressiveWrapper

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

class PositionalEncoding(nn.Module):

   def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
       super(PositionalEncoding, self).__init__()
       self.device = device
       self.dropout = nn.Dropout(p=dropout)
       self.lpe = nn.Embedding(max_len+1, d_model)
       # self.weight = None
       self.indices = torch.arange(max_len).unsqueeze(1) + 1
       if device is not None:
           self.indices = self.indices.to(self.device)

   def init_weights(self):
       initrange = 0.1
       self.lpe.weight.data.uniform_(-initrange, initrange)

   def forward(self, x, indices = None):
       np.save(str(uuid.uuid4())+".np",self.lpe.weight.data.cpu().numpy())
       if indices is None:
           indices = self.indices[:x.size(0),:]
           indices = self.dropout(indices)
       x = x + self.lpe(indices)
       return self.dropout(x)


class LearnedPositionalEncoding(nn.Module): # emm this isn't learned lol
    def __init__(self, d_model, input_length, dropout=0.1, device=None):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = nn.Parameter((torch.zeros(input_length, 1, d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class BasicTransformerModel(nn.Module):
    def __init__(self, dout, dinp, nhead, dhid, nlayers, dropout=0.5, ntokens=0, device=None, use_pos_emb=False, use_rel_pos_emb=False, input_length=0,use_x_transformers=False, opt=None, discrete_inputs=False):
        super(BasicTransformerModel, self).__init__()
        self.device = device
        self.model_type = 'Transformer'
        self.use_x_transformers = use_x_transformers
        self.discrete_inputs = discrete_inputs
        if not use_x_transformers:
            if discrete_inputs:
                assert ntokens != 0 #ntoken needs to be set if we are to use an embedding layer (discrete inputs)
                self.encoder = nn.Embedding(ntokens, dinp)
            self.encoder1 = nn.Linear(dinp, dhid)
            if use_pos_emb:
                self.pos_encoder = LearnedPositionalEncoding(dhid, input_length, dropout)
            encoder_layers = TransformerEncoderLayer(dhid, nhead, dhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.dinp = dinp
            self.dhid = dhid
            self.decoder = nn.Linear(dhid, dout)
            self.use_pos_emb = use_pos_emb
            self.use_rel_pos_emb = use_rel_pos_emb
            self.input_length = input_length
            if use_rel_pos_emb and input_length > 1:
                #assert input_length > 0
                self.pos_emb = nn.Parameter((torch.zeros(input_length, input_length)))
                # self.pos_emb = nn.Parameter((torch.eye(input_length, input_length)))
                # self.pos_emb = nn.Parameter((torch.randn(input_length, input_length))/np.sqrt(dinp))

            self.init_weights()
            #self.pos_encoder.init_weights()
        else:
            if discrete_inputs:
                assert ntoken is not None #ntoken needs to be set if we are to use an embedding layer (discrete inputs)
                self.encoder = nn.Embedding(ntoken, dinp)
            self.model = ContinuousTransformerWrapper(
                dim_in = dinp,
                dim_out = dout,
                max_seq_len = 1024,
                use_pos_emb = use_pos_emb,
                attn_layers = Encoder(
                    dim = dhid,
                    depth = nlayers,
                    heads = nhead,
                    rotary_pos_emb = opt.use_rotary_pos_emb,
                    #rel_pos_bias = True
                )
            )

    def generate_square_subsequent_mask(self, sz, prefix_length = 1):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask[:,:prefix_length] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        #print(src.shape)
        if self.discrete_inputs:
            src = self.encoder(src.squeeze(0))
        if not self.use_x_transformers:
            src = self.encoder1(src)
            # import pdb;pdb.set_trace()
            #src *= math.sqrt(self.dhid)
            if self.use_pos_emb:
                src = self.pos_encoder(src)
            #src /= math.sqrt(self.dhid)
            # print(src)
            # print(torch.mm(src[:,0,:],src[:,0,:].T))
            if self.use_rel_pos_emb and self.input_length > 1:
                #print(self.pos_emb)
                #src_mask += self.pos_emb
                if src_mask is not None:
                    output = self.transformer_encoder(src, src_mask + self.pos_emb)
                else:
                    output = self.transformer_encoder(src, self.pos_emb)
                #output = self.transformer_encoder(src, self.pos_emb)
            else:
                if src_mask is not None:
                    output = self.transformer_encoder(src, src_mask)
                else:
                    output = self.transformer_encoder(src)
                #output = self.transformer_encoder(src)
            output = self.decoder(output)
            #print(output.shape)
            if self.discrete_inputs:
                #return output.unsqueeze(0)
                return output.permute(1,0,2)
            else:
                return output
        else:
            assert src_mask == None
            src = src.permute(1,0,2)
            mask = torch.ones(src.shape[0], src.shape[1]).bool().cuda()
            output = self.model(src, mask = mask)
            # output = self.model(src.permute(1,0,2))
            return output.permute(1,0,2)

class BasicTransformerModelCausal(nn.Module):
    def __init__(self, dout, dinp, nhead, dhid, nlayers, dropout=0.5, ntokens=0, device=None, use_pos_emb=False, use_rel_pos_emb=False, input_length=0,use_x_transformers=False, opt=None, discrete_inputs=False):
        self.model = BasicTransformerModel(self, dout, dinp, nhead, dhid, nlayers, dropout=0.5, ntokens=0, device=None, use_pos_emb=False, use_rel_pos_emb=False, input_length=0,use_x_transformers=False, opt=None, discrete_inputs=False)
        self.mask = self.model.generate_square_subsequent_mask(input_length)
    def init_weights(self):
        self.model.init_weights()
    def generate_square_subsequent_mask(self, sz, prefix_length = 1):
        self.model.generate_square_subsequent_mask(sz,prefix_length)
    def forward(self, src, src_mask=None):
        return self.model(src,src_mask=self.mask)


class EncDecTransformerModel(nn.Module):

    def __init__(self, dout, src_d, tgt_d, nhead, dhid, nlayers, dropout=0.5,device=None,use_pos_emb=False,src_length=0,tgt_length=0,use_x_transformers=False,opt=None):
        super(EncDecTransformerModel, self).__init__()
        self.device = device
        self.model_type = 'Transformer'
        self.use_x_transformers = use_x_transformers
        self.encoder1 = nn.Linear(src_d, dhid)
        self.encoder2 = nn.Linear(tgt_d, dhid)
        if not use_x_transformers:
            self.transformer = Transformer(d_model=dhid, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dropout=0, activation="relu")
            #enc_layer = TransformerEncoderLayer(d_model=dhid, nhead=nhead, dropout=0, activation="relu")
            #self.transformerEnc = TransformerEncoder(enc_layer, nlayers)
        else:
            self.transformer = EncDecXTransformer(enc_dim_in=src_d, enc_dim_out=tgt_d, dec_din_in=tgt_d, edec_dim_out=dout, enc_dim=dhid, dec_dim=dhid, nc_heads=nhead, dec_heads=nhead, enc_depth=nlayers, dec_depth=nlayers, enc_dropout=dropout, dec_dropout=dropout, enc_max_seq_len=1024, dec_max_seq_len=1024)

        #xdecoder = Decoder(dim=dhid, depth=nlayers, heads=nhead, cross_attend=True)
        #self.transformer = Transformer(d_model=dhid, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dropout=dropout, activation="gelu", custom_decoder=xdecoder)
        #self.transformer = Transformer(d_model=dhid, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers)
        # self.encoder = nn.Embedding(ntoken, dinp)
        self.src_d = src_d
        self.tgt_d = tgt_d
        self.dhid = dhid
        self.decoder = nn.Linear(dhid, dout)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            assert src_length > 0
            assert tgt_length > 0
            self.src_pos_emb = nn.Parameter((torch.zeros(src_length, src_length)))
            self.tgt_pos_emb = nn.Parameter((torch.zeros(tgt_length, tgt_length)))

        if not use_x_transformers:
            tgt_mask = self.generate_square_subsequent_mask(tgt_length)
        else:
            tgt_mask = self.generate_square_subsequent_mask_bool(tgt_length)
        self.register_buffer("tgt_mask", tgt_mask)
        #a = torch.randn(32,3,512)
        #b = torch.randn(32,3,512)
        #self.register_buffer('a', a)
        #self.register_buffer('b', b)


        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_subsequent_mask_bool(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).bool()
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        if not self.use_x_transformers:
            # import pdb;pdb.set_trace()
            src = self.encoder1(src)
            tgt = self.encoder2(tgt)
            tgt_mask = self.tgt_mask[:tgt.shape[0], :tgt.shape[0]]
            if self.use_pos_emb:
                tgt_pos_emb = self.tgt_pos_emb[:tgt.shape[0], :tgt.shape[0]]
                # import pdb;pdb.set_trace()
                output = self.transformer(src=src, tgt=tgt, src_mask=self.src_pos_emb, tgt_mask=tgt_pos_emb+tgt_mask)
                #output = self.transformer(src=src, tgt=tgt)
            else:
                output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
            #output = self.transformer(src=self.a, tgt=self.b)
            #output = self.transformerEnc(src=self.a)
            output = self.decoder(output)
            return output
        else:
            src = self.encoder1(src)
            tgt = self.encoder2(tgt)
            #tgt_mask = self.tgt_mask[:tgt.shape[0], :tgt.shape[0]]
            # if self.use_pos_emb:
            #     tgt_pos_emb = self.tgt_pos_emb[:tgt.shape[0], :tgt.shape[0]]
            #     # import pdb;pdb.set_trace()
            #     output = self.transformer(src=src.permute(1,2,0), tgt=tgt.permute(1,2,0), src_mask=self.src_pos_emb, tgt_mask=tgt_pos_emb+tgt_mask)
            #     #output = self.transformer(src=src, tgt=tgt)
            # else:
            # output = self.transformer(src=src.permute(1,0,2), tgt=tgt.permute(1,0,2), tgt_mask=tgt_mask)
            # output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
            output = self.transformer(src=src, tgt=tgt)
            #output = self.transformer(src=self.a, tgt=self.b)
            #output = self.transformer(src=tgt, tgt=tgt) #hmm thats an interesting way of residual attention
            output = self.decoder(output)
            return output
            #return


class EncDecXTransformer(nn.Module):
    def __init__(
            self,
            *,
            # dim,
            tie_token_emb = False,
            **kwargs
    ):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        # import pdb;pdb.set_trace()
        # assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        # enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs = pick_and_pop(['max_seq_len'], enc_kwargs)
        # enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)

        # dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs = pick_and_pop(['max_seq_len'], dec_kwargs)

        self.encoder = ContinuousTransformerWrapper(
            **enc_transformer_kwargs,
            attn_layers = Encoder(**enc_kwargs)
        )

        self.decoder = ContinuousTransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(cross_attend = True, **dec_kwargs)
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        # self.decoder = AutoregressiveWrapper(self.decoder)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, src_mask = None):
        encodings = self.encoder(seq_in, return_embeddings = True, mask = src_mask)
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = src_mask)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        enc = self.encoder(src, mask = src_mask, return_embeddings = True)
        #out = self.decoder(tgt, context = enc, mask = tgt_mask, context_mask = src_mask)
        out = self.decoder(tgt, context = enc)
        return out
