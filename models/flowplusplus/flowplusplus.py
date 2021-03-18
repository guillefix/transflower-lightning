import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.flowplusplus.act_norm import ActNorm
from models.flowplusplus.inv_conv import InvConv, InvertibleConv1x1
from models.flowplusplus.nn import GatedConv
from models.flowplusplus.coupling import Coupling
from models.util import channelwise, checkerboard, Flip, safe_log, squeeze, unsqueeze


class FlowPlusPlus(nn.Module):
    """Flow++ Model

    Based on the paper:
    "Flow++: Improving Flow-Based Generative Models
        with Variational Dequantization and Architecture Design"
    by Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel
    (https://openreview.net/forum?id=Hyg74h05tX).

    Args:
        scales (tuple or list): Number of each type of coupling layer in each
            scale. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_dequant_blocks (int): Number of blocks in the dequantization flows.
    """
    def __init__(self,
                 scales=((0, 4), (2, 3)),
                 in_shape=(3, 32, 32),
                 cond_dim=0,
                 mid_channels=96,
                 num_blocks=10,
                 num_components=32,
                 use_attn=True,
                 use_logmix=True,
                 drop_prob=0.2):
        super(FlowPlusPlus, self).__init__()
        # Register bounds to pre-process images, not learnable
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _FlowStep(scales=scales,
                               in_shape=in_shape,
                               cond_dim=cond_dim,
                               mid_channels=mid_channels,
                               num_blocks=num_blocks,
                               num_components=num_components,
                               use_attn=use_attn,
                               use_logmix=use_logmix,
                               drop_prob=drop_prob)

    def forward(self, x, cond, reverse=False):
        if cond is not None:
            cond = cond.permute(0,2,1).unsqueeze(3)
            
        if not reverse:        
            if x is not None:
                x = x.permute(0,2,1).unsqueeze(3)
        else:
            c, h, w = self.flows.z_dim()
            x = torch.randn((cond.size(0), c, h, w), dtype=torch.float32).type_as(cond)
            
        sldj = torch.zeros(x.size(0), device=x.device)
        x, sldj = self.flows(x, cond, sldj, reverse)
        
        if reverse:
            if x is not None:
                x = x.squeeze(3).permute(0,2,1)

        return x, sldj

    def loss_generative(self, z, sldj):
        """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

        Args:
            k (int or float): Number of discrete values in each input dimension.
                E.g., `k` is 256 for natural images.

        See Also:
            Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
        """
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1)# \
#            - np.log(k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()/float(np.log(2.) * z.size(2) * z.size(3))

        return nll
        
class _FlowStep(nn.Module):
    """Recursive builder for a Flow++ model.

    Each `_FlowStep` corresponds to a single scale in Flow++.
    The constructor is recursively called to build a full model.

    Args:
        scales (tuple): Number of each type of coupling layer in each scale.
            Each scale is a 2-tuple of the form (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_components (int): Number of components in the mixture.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, scales, in_shape, cond_dim, mid_channels, num_blocks, num_components, use_attn, use_logmix, drop_prob):
        super(_FlowStep, self).__init__()
        in_channels, in_height, in_width = in_shape
        num_channelwise, num_checkerboard = scales[0]
        #import pdb;pdb.set_trace()
        channels = []
        for i in range(num_channelwise):
            new_channels = in_channels// 2
            out_channels = in_channels-new_channels
            channels += [InvertibleConv1x1(in_channels),
                         Coupling(in_channels=new_channels + cond_dim,
                                  out_channels=out_channels,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  use_logmix=use_logmix,
                                  drop_prob=drop_prob)]#,
                         #Flip()] Flip currently does not work with odd number of channels. But is it needed when we have channel mixing with 1x1convs? 

        checkers = []
        for i in range(num_checkerboard):
            checkers += [InvertibleConv1x1(in_channels),
                         Coupling(in_channels=in_channels+cond_dim,
                                  out_channels=in_channels,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  use_logmix=use_logmix,
                                  drop_prob=drop_prob)]#,
                         #Flip()]
        self.channels = nn.ModuleList(channels) if channels else None
        self.checkers = nn.ModuleList(checkers) if checkers else None

        if len(scales) <= 1:
            self.next = None
        else:
            next_shape = (in_channels, in_height // 2, in_width)
            self.next = _FlowStep(scales=scales[1:],
                                  in_shape=next_shape,
                                  cond_dim=2*cond_dim,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  use_logmix=use_logmix,
                                  drop_prob=drop_prob)
                                  
        self.z_shape = (in_channels, in_height, in_width)
        
    def z_dim(self):
        return self.z_shape

    def forward(self, x, cond, sldj, reverse=False):
            
        if reverse:
            #import pdb;pdb.set_trace()
            if self.next is not None:
                x = squeeze(x)
                cond = squeeze(cond)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, cond, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)
                cond = unsqueeze(cond)

            if self.checkers:
                x = checkerboard(x)
                for flow in reversed(self.checkers):
                    x, sldj = flow(x, cond, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.channels:
                x = channelwise(x)
                for flow in reversed(self.channels):
                    x, sldj = flow(x, cond, sldj, reverse)
                x = channelwise(x, reverse=True)
        else:
            # import pdb;pdb.set_trace()
            if self.channels:
                x = channelwise(x)
                for flow in self.channels:
                    #import pdb;pdb.set_trace()
                    x, sldj = flow(x, cond, sldj, reverse)
                x = channelwise(x, reverse=True)

            if self.checkers:
                x = checkerboard(x)
                for flow in self.checkers:
                    x, sldj = flow(x, cond, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.next is not None:
                # import pdb;pdb.set_trace()
                # here we apply the flow steps but only to dimensions sampled at a lower scale. Hmm feels a bit weird
                x = squeeze(x)
                cond = squeeze(cond)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, cond, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)

        return x, sldj
        
