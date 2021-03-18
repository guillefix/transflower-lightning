import math
import torch
import torch.nn as nn

from models.flowplusplus import log_dist as logistic
from models.flowplusplus.nn import NN


class Coupling(nn.Module):
    """Mixture-of-Logistics Coupling layer in Flow++

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the transformation network.
        num_blocks (int): Number of residual blocks in the transformation network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in the NN blocks.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, in_channels, out_channels, mid_channels, num_blocks, num_components, drop_prob,
                 use_attn=True, use_logmix=True, aux_channels=None):
        super(Coupling, self).__init__()
        
        self.nn = NN(in_channels, out_channels, mid_channels, num_blocks, num_components, drop_prob, use_attn, aux_channels)
        self.use_logmix = use_logmix
        self.offset = 2.0
        self.sigmoid_offset = 1 - 1 / (1 + math.exp(-self.offset))

    def forward(self, x, cond, sldj=None, reverse=False, aux=None):
        x_change, x_id = x
        #import pdb;pdb.set_trace()

        x_id_cond = torch.cat((x_id, cond), dim=1)
        a, b, pi, mu, s = self.nn(x_id_cond, aux)
        scale = (torch.sigmoid(a+self.offset)+self.sigmoid_offset)

        if reverse:
            out = x_change / scale - b
            if self.use_logmix:
                out, scale_ldj = logistic.inverse(out, reverse=True)
                #out = out.clamp(1e-5, 1. - 1e-5)
                out = logistic.mixture_inv_cdf(out, pi, mu, s)
                logistic_ldj = logistic.mixture_log_pdf(out, pi, mu, s)
                sldj = sldj - (torch.log(scale) + scale_ldj + logistic_ldj).flatten(1).sum(-1)
            else:
                sldj = sldj - torch.log(scale).flatten(1).sum(-1)
        else:
            if self.use_logmix:
                out = logistic.mixture_log_cdf(x_change, pi, mu, s).exp()
                out, scale_ldj = logistic.inverse(out)
                logistic_ldj = logistic.mixture_log_pdf(x_change, pi, mu, s)
                sldj = sldj + (logistic_ldj + scale_ldj + torch.log(scale)).flatten(1).sum(-1)
            else:
                out = x_change
                sldj = sldj + torch.log(scale).flatten(1).sum(-1)
            
            out = (out + b) * scale
            

        x = (out, x_id)

        return x, sldj
