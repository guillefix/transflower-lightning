import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm
from models.util import concat_elu #,WNConv2d
from models.transformer import BasicTransformerModel

class TransformerNN(nn.Module):
    """Neural network used to parametrize the transformations of an MLCoupling.

    An `NN` is a stack of blocks, where each block consists of the following
    two layers connected in a residual fashion:
      1. Conv: input -> nonlinearit -> conv3x3 -> nonlinearity -> gate
      2. Attn: input -> conv1x1 -> multihead self-attention -> gate,
    where gate refers to a 1×1 convolution that doubles the number of channels,
    followed by a gated linear unit (Dauphin et al., 2016).
    The convolutional layer is identical to the one used by PixelCNN++
    (Salimans et al., 2017), and the multi-head self attention mechanism we
    use is identical to the one in the Transformer (Vaswani et al., 2017).

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        num_channels (int): Number of channels in each block of the network.
        num_blocks (int): Number of blocks in the network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in each block.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, in_channels, out_channels, num_channels, num_layers, num_heads, num_components, drop_prob, use_pos_emb, use_rel_pos_emb, input_length, concat_dims, output_length):
        #import pdb;pdb.set_trace()
        super(TransformerNN, self).__init__()
        self.k = num_components  # k = number of mixture components
        # import pdb;pdb.set_trace()
        self.transformer = BasicTransformerModel(out_channels * (2 + 3 * self.k), in_channels, num_heads, num_channels, num_layers, drop_prob, use_pos_emb=use_pos_emb, use_rel_pos_emb=use_rel_pos_emb, input_length=input_length)
        self.rescale = weight_norm(Rescale(out_channels))
        self.out_channels = out_channels
        self.concat_dims = concat_dims
        self.output_length = output_length

    def forward(self, x, aux=None):
        b, c, h, w = x.size()
        # import pdb;pdb.set_trace()
        x = x.squeeze(-1) # only squeeze the w dimension (important coz otherwise it would squeeze batch dim if theres only one element in minibatch..
        # import pdb;pdb.set_trace()
        x = x.permute(2,0,1)
        # import pdb;pdb.set_trace()
        if self.concat_dims:
            x = self.transformer(x)
            # x = torch.mean(self.transformer(x), dim=0, keepdim=True)
            # x = 0.5*x + 0.5*torch.mean(x, dim=0, keepdim=True)
            # x = self.transformer(x)[:1]
        else:
            x = self.transformer(x)[:self.output_length]
        # import pdb;pdb.set_trace()
        x = x.permute(1,2,0)
        # Split into components and post-process
        if self.concat_dims:
            x = x.view(b, -1, self.out_channels, h, w)
            # x = x.view(b, -1, self.out_channels, 1, w)
        else:
            x = x.view(b, -1, self.out_channels, self.output_length, w)
        s, t, pi, mu, scales = x.split((1, 1, self.k, self.k, self.k), dim=1)
        s = self.rescale(torch.tanh(s.squeeze(1)))
        t = t.squeeze(1)
        scales = scales.clamp(min=-7)  # From the code in original Flow++ paper

        return s, t, pi, mu, scales

class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
