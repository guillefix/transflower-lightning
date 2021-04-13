import torch
import torch.nn as nn

from models.util import mean_dim


class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width):
        super(_BaseNorm, self).__init__()

        # Input gets concatenated along channel axis
        #num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            mean, inv_std = self._get_moments(x)
            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, cond, ldj=None, reverse=False):
        #import pdb;pdb.set_trace()
        x = torch.cat(x, dim=1)
        # import pdb;pdb.set_trace()
        if not self.is_initialized:
            print("Initializing norm Layer!")
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        x = x.chunk(2, dim=1)

        return x, ldj


class BatchNorm(nn.Module):
    def __init__(self, num_channels, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_channels, 1, 1))
        self.eps = 1e-5
        self.momentum = momentum
        self.inv_std = None
        self.register_buffer('is_initialized', torch.zeros(1))

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True).detach()
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True).detach()
        # inv_std = 1. / (var.sqrt() + self.eps)
        if not self.is_initialized:
            self.running_mean.data.copy_(mean.data)
            self.running_var.data.copy_(var.data)
            self.is_initialized += 1.
        else:
            if self.momentum < 1.0:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                self.running_mean.data.copy_(mean.data)
                self.running_var.data.copy_(var.data)

    def forward(self, x, cond, ldj=None, reverse=False):
        # import pdb;pdb.set_trace()
        x = torch.cat(x, dim=1)
        if self.training:
            # print("HI")
            self._get_moments(x)
        # print(self.running_var[0])
        inv_std = 1. / (self.running_var.sqrt() + self.eps)
        if reverse:
            x = self._center(x, self.beta, reverse)
            x, ldj = self._scale(x, ldj, self.gamma, reverse)
            x, ldj = self._scale(x, ldj, inv_std, reverse)
            x = self._center(x, self.running_mean, reverse)
        else:
            x = self._center(x, self.running_mean, reverse)
            x, ldj = self._scale(x, ldj, inv_std, reverse)
            x, ldj = self._scale(x, ldj, self.gamma, reverse)
            x = self._center(x, self.beta, reverse)
        x = x.chunk(2, dim=1)

        return x, ldj

    def _center(self, x, centerer, reverse=False):
        if reverse:
            return x + centerer
        else:
            return x - centerer

    def _scale(self, x, sldj, scaler, reverse=False):
        if reverse:
            x = x / scaler
            sldj = sldj - scaler.log().sum() * x.size(2) * x.size(3)
        else:
            x = x * scaler
            sldj = sldj + scaler.log().sum() * x.size(2) * x.size(3)

        return x, sldj

class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels):
        super(ActNorm, self).__init__(num_channels, 1, 1)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum() * x.size(2) * x.size(3)
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum() * x.size(2) * x.size(3)

        return x, sldj


class PixNorm(_BaseNorm):
    """Pixel-wise Activation Normalization used in Flow++

    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.
    """
    def _get_moments(self, x):
        mean = torch.mean(x.clone(), dim=0, keepdim=True)
        var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum()
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum()

        return x, sldj
