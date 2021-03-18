import torch
import torch.nn as nn
import torch.nn.functional as F


class Flip(nn.Module):
    def forward(self, x, cond, sldj, reverse=False):
        assert isinstance(x, tuple) and len(x) == 2
        return (x[1], x[0]), sldj


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.

    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.

    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def checkerboard(x, reverse=False):
    """Split x in a checkerboard pattern. Collapse horizontally."""
    # Get dimensions
    if reverse:
        b, c, h, w = x[0].size()
        w *= 2
        device = x[0].device
    else:
        b, c, h, w = x.size()
        device = x.device

    # Get list of indices in alternating checkerboard pattern
    y_idx = []
    z_idx = []
    for i in range(h):
        for j in range(w):
            if (i % 2) == (j % 2):
                y_idx.append(i * w + j)
            else:
                z_idx.append(i * w + j)
    y_idx = torch.tensor(y_idx, dtype=torch.int64, device=device)
    z_idx = torch.tensor(z_idx, dtype=torch.int64, device=device)

    if reverse:
        y, z = (t.contiguous().view(b, c, h // 2 * w) for t in x)
        x = torch.zeros(b, c, h * w, dtype=y.dtype, device=y.device)
        x[:, :, y_idx] += y
        x[:, :, z_idx] += z
        x = x.view(b, c, h, w)

        return x
    else:
        if h % 2 != 0:
            raise RuntimeError('Checkerboard got odd height input: {}'.format(h))

        x = x.view(b, c, h * w)
        y = x[:, :, y_idx].view(b, c, h // 2, w)
        z = x[:, :, z_idx].view(b, c, h // 2, w)

        return y, z


def channelwise(x, reverse=False):
    """Split x channel-wise."""
    if reverse:
        x = torch.cat(x, dim=1)
        return x
    else:
        y, z = x.chunk(2, dim=1)
        return y, z


def squeeze(x):
    """Trade spatial extent for channels. I.e., convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    # import pdb; pdb.set_trace()
    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w, 1)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 2, h // 2, w)

    return x


def unsqueeze(x):
    """Trade channels channels for spatial extent. I.e., convert each
    4x1x1 volume of input into a 1x4x4 volume of output.

    Args:
        x (torch.Tensor): Input to unsqueeze.

    Returns:
        x (torch.Tensor): Unsqueezed tensor.
    """
    b, c, h, w = x.size()
    x = x.view(b, c // 2, 2, 1, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 2, h * 2, w)

    return x


def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(torch.cat((x, -x), dim=1))


def safe_log(x):
    return torch.log(x.clamp(min=1e-22))
