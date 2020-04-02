import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()

        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            x = torch.nn.functional.elu(x)
        return x

    def __call__(self, *args, **kwargs):
        return super(Activation, self).__call__(*args, **kwargs)


class MyConv2d(nn.Module):
    """
    Class to encapsulate conv2d with custom initialization as in:
      https://github.com/CompVis/vunet/blob/db88509b867c9472a1e67b19f41720e1132c6a8c/nn.py#L46-L66
    """
    def __init__(self, c_in, c_out, kernel_size, stride, padding, w_norm: bool):
        super(MyConv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=True)
        if w_norm:
            # Weight normalization, see: https://arxiv.org/pdf/1602.07868.pdf
            self.conv = weight_norm(self.conv, dim=0)

    def forward(self, x, skip_in=None):
        if skip_in is not None:
            x = torch.cat([x, skip_in], dim=1)
        return self.conv(x)

    def __call__(self, *args, **kwargs):
        return super(MyConv2d, self).__call__(*args, **kwargs)


class NiN(nn.Module):
    """
    Encoder-Decoder skip connection
    in_c == out_c and always kernel_size equals 1
    """

    def __init__(self, c_in, c_out, w_norm):
        super(NiN, self).__init__()
        self.layers = nn.Sequential(Activation('elu'),
                                    MyConv2d(c_in=c_in, c_out=c_out, kernel_size=1, stride=1,
                                             padding=0, w_norm=w_norm))

    def forward(self, x):
        return self.layers(x)

    def __call__(self, *args, **kwargs):
        return super(NiN, self).__call__(*args, **kwargs)


class DeConv2d(nn.Module):
    """
    Class to encapsulate conv2d with custom initialization as in:
      https://github.com/CompVis/vunet/blob/db88509b867c9472a1e67b19f41720e1132c6a8c/nn.py#L46-L66
    """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, w_norm: bool):
        super(DeConv2d, self).__init__()
        # todo: magic init
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=True,
                                       output_padding=1)
        if w_norm:
            # Weight normalization, see: https://arxiv.org/pdf/1602.07868.pdf
            self.conv = weight_norm(self.conv, dim=1)

    def forward(self, x):
        return self.conv(x)

    def __call__(self, *args, **kwargs):
        return super(DeConv2d, self).__call__(*args, **kwargs)


class Residual(nn.Module):
    def __init__(self, c_in, c_out, activation, drop_prob, w_norm, use_sampling=False):
        # todo: we are ignoring gating on purpose
        # todo: handle skip connection
        # todo: we implemented NiN as Conv1x1
        super(Residual, self).__init__()

        self.use_sampling = use_sampling
        activate = Activation(activation)
        dropout = torch.nn.Dropout2d(drop_prob, inplace=False)

        conv3x3 = MyConv2d(c_in, c_out=c_out, kernel_size=3, stride=1, padding=1, w_norm=w_norm)

        self.layers = torch.nn.Sequential(activate, dropout, conv3x3)

    def forward(self, x, skip_in):
        residual = x
        if skip_in is not None:
            x = torch.cat([residual, skip_in], dim=1)
        return self.layers(x) + residual

    def __call__(self, *args, **kwargs):
        return super(Residual, self).__call__(*args, **kwargs)


class DownSample(nn.Module):
    def __init__(self, c_in, c_out, w_norm):
        super(DownSample, self).__init__()

        self.down = MyConv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, w_norm=w_norm)

    def forward(self, x, skip_in):
        return self.down(x)

    def __call__(self, *args, **kwargs):
        return super(DownSample, self).__call__(*args, **kwargs)


class UpSample(nn.Module):
    """
    See: https://github.com/CompVis/vunet/blob/db88509b867c9472a1e67b19f41720e1132c6a8c/nn.py#L123-L129
    """
    def __init__(self, c_in, c_out, w_norm, mode):
        super(UpSample, self).__init__()

        self.mode = mode

        if self.mode == 'subpixel':
            self.depth4x = MyConv2d(c_in, c_out=4 * c_out, kernel_size=3, stride=1, padding=1, w_norm=w_norm)
            self.depth2space = DepthToSpace(block_size=2)
        elif self.mode == 'conv2d_t':
            self.up = DeConv2d(c_in, c_out=c_out, kernel_size=3, stride=2, padding=1, w_norm=w_norm)
        elif self.mode == 'nearest':
            self.conv = MyConv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, w_norm=w_norm)
        else:
            raise ValueError(f'Unknown mode: {self.mode}.')

    def forward(self, x, skip_in):
        if skip_in is not None:
            x = torch.cat([x, skip_in], dim=1)

        if self.mode == 'subpixel':
            x = self.depth4x(x)
            out = self.depth2space(x)
        elif self.mode == 'conv2d_t':
            out = self.up(x)
        elif self.mode == 'nearest':
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            out = self.conv(x)
        return out

    def __call__(self, *args, **kwargs):
        return super(UpSample, self).__call__(*args, **kwargs)


class Sampler(nn.Module):
    def __init__(self, c_in, c_out, w_norm):
        super(Sampler, self).__init__()
        self.conv = MyConv2d(c_in=c_in, c_out=c_out, kernel_size=3, stride=1, padding=1, w_norm=w_norm)

    def forward(self, x: torch.Tensor, cov: float = 1.0):
        mu = self.conv(x)
        # todo move sampling in GPU
        sample = mu + torch.randn(*mu.size()).to(mu.device) * cov
        return mu, sample

    def __call__(self, *args, **kwargs):
        return super(Sampler, self).__call__(*args, **kwargs)


class DepthToSpace(nn.Module):
    """
    https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec
    """
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def __call__(self, *args, **kwargs):
        return super(DepthToSpace, self).__call__(*args, **kwargs)


class SpaceToDepth(nn.Module):
    """
    https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec
    """
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

    def __call__(self, *args, **kwargs):
        return super(SpaceToDepth, self).__call__(*args, **kwargs)


class RGB2YCbCr(nn.Module):
    """
    Apply color conversion to a BGR 4D tensor in [-1,1]
    """
    def __init__(self):
        super(RGB2YCbCr, self).__init__()

    def forward(self, input):
        input = (input + 1) / 2
        B = input[:, 0, ...]
        G = input[:, 1, ...]
        R = input[:, 2, ...]

        Y = 16.0 + 65.481 * R + 128.553 * G + 24.966 * B
        Cb = 128 - 37.797 * R - 74.203 * G + 112 * B
        Cr = 128 + 112 * R - 93.786 * G - 18.214 * B
        return torch.stack([Y, Cb, Cr], 1)

    def __call__(self, *args, **kwargs):
        return super(RGB2YCbCr, self).__call__(*args, **kwargs)