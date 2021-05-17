import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.linalg

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            # import pdb;pdb.set_trace()
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)
            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.first_pass = True
        self.saved_weight = None
        self.saved_dsldj = None

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            dlogdet = torch.slogdet(self.weight)[1] * input.size(2) * input.size(3)
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = self.log_s.sum() * input.size(2) * input.size(3)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, x, cond, sldj=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        x = torch.cat(x, dim=1)
        if not reverse:
            weight, dsldj = self.get_weight(x, reverse)
        else:
            if self.first_pass:
                weight, dsldj = self.get_weight(x, reverse)
                self.saved_weight = weight
                if sldj is not None:
                    self.saved_dsldj = dsldj
                self.first_pass = False
            else:
                weight = self.saved_weight
                if sldj is not None:
                    dsldj = self.saved_dsldj

        if not reverse:
            x = F.conv2d(x, weight)
            if sldj is not None:
                sldj = sldj + dsldj
        else:
            x = F.conv2d(x, weight)
            if sldj is not None:
                sldj = sldj - dsldj
        x = x.chunk(2, dim=1)
        return x, sldj


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
        random_init (bool): Initialize with a random orthogonal matrix.
            Otherwise initialize with noisy identity.
    """
    def __init__(self, num_channels, random_init=False):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        if random_init:
            # Initialize with a random orthogonal matrix
            w_init = np.random.randn(self.num_channels, self.num_channels)
            w_init = np.linalg.qr(w_init)[0]
        else:
            # Initialize as identity permutation with some noise
            w_init = np.eye(self.num_channels, self.num_channels) \
                     + 1e-3 * np.random.randn(self.num_channels, self.num_channels)
        self.weight = nn.Parameter(torch.from_numpy(w_init.astype(np.float32)))

    def forward(self, x, cond, sldj, reverse=False):
        x = torch.cat(x, dim=1)

        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        x = F.conv2d(x, weight)
        x = x.chunk(2, dim=1)

        return x, sldj
