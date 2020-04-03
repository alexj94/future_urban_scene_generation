import argparse

import torch
import torch.nn as nn

from vunet.layers import Activation
from vunet.layers import DepthToSpace
from vunet.layers import DownSample
from vunet.layers import MyConv2d
from vunet.layers import NiN
from vunet.layers import Residual
from vunet.layers import Sampler
from vunet.layers import SpaceToDepth
from vunet.layers import UpSample


class AutoRegressiveBlock(nn.Module):
    def __init__(self, drop_prob, w_norm):
        super(AutoRegressiveBlock, self).__init__()
        # conversion objects
        self.s2d = SpaceToDepth(2)
        self.d2s = DepthToSpace(2)

        self.w_norm = w_norm
        self.drop_prob = drop_prob
        # first residual outside autoregression
        self.residual_init = Residual(c_in=256, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                      w_norm=self.w_norm)
        self.sampler_0 = Sampler(c_in=512, c_out=128, w_norm=self.w_norm)
        self.residual_0 = Residual(c_in=512 + 512, c_out=512, activation=Activation('elu'),
                                   drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.sampler_1 = Sampler(c_in=512, c_out=128, w_norm=self.w_norm)
        self.residual_1 = Residual(c_in=512 + 512, c_out=512, activation=Activation('elu'),
                                   drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.sampler_2 = Sampler(c_in=512, c_out=128, w_norm=self.w_norm)
        # residual applied to s2d
        self.residual_2 = Residual(c_in=512 + 512, c_out=512, activation=Activation('elu'),
                                   drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.sampler_3 = Sampler(c_in=512, c_out=128, w_norm=self.w_norm)
        # 0 sampler
        self.nin_0 = NiN(c_in=128, c_out=512, w_norm=self.w_norm)

        # 1 sampler
        self.nin_1 = NiN(c_in=128, c_out=512, w_norm=self.w_norm)
        # 2 sampler
        self.nin_2 = NiN(c_in=128, c_out=512, w_norm=self.w_norm)
        self.residual_s2d = Residual(c_in=128, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                     w_norm=self.w_norm)

        # 3 sampler
        # note there is no residual here, we already have 4 means and samples, no need to compute a x_ nobody will use

    def forward(self, x, skip_a, enc_down_mu=None):

        x = self.residual_init(x, skip_in=skip_a)
        x_ = self.s2d(self.residual_s2d(x, skip_in=None))
        if enc_down_mu is not None:
            g_0, g_1, g_2, g_3 = torch.split(self.s2d(enc_down_mu), 128, 1)
            g_0 = self.nin_0(g_0)
            g_1 = self.nin_1(g_1)
            g_2 = self.nin_2(g_2)

        mu_0_0, z_0_0 = self.sampler_0(x_)
        if enc_down_mu is not None:
            x_ = self.residual_0(x_, skip_in=g_0)
        else:
            x_ = self.residual_0(x_, skip_in=self.nin_0(z_0_0))
        mu_0_1, z_0_1 = self.sampler_1(x_)
        if enc_down_mu is not None:
            x_ = self.residual_1(x_, skip_in=g_1)
        else:
            x_ = self.residual_1(x_, skip_in=self.nin_1(z_0_1))
        mu_0_2, z_0_2 = self.sampler_2(x_)
        if enc_down_mu is not None:
            x_ = self.residual_2(x_, skip_in=g_2)
        else:
            x_ = self.residual_2(x_, skip_in=self.nin_2(z_0_2))
        mu_0_3, z_0_3 = self.sampler_3(x_)

        mu_0 = self.d2s(torch.cat([mu_0_0, mu_0_1, mu_0_2, mu_0_3], 1)).contiguous()
        z_0 = self.d2s(torch.cat([z_0_0, z_0_1, z_0_2, z_0_3], 1))

        return x, mu_0, z_0

    def __call__(self, *args, **kwargs):
        return super(AutoRegressiveBlock, self).__call__(*args, **kwargs)


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, drop_prob, w_norm):
        super(DownBlock, self).__init__()

        self.w_norm = w_norm
        self.drop_prob = drop_prob

        self.down = DownSample(c_in=c_in, c_out=c_out, w_norm=self.w_norm)
        self.residual_0 = Residual(c_in=c_out, c_out=c_out, activation=Activation('elu'), drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.residual_1 = Residual(c_in=c_out, c_out=c_out, activation=Activation('elu'), drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)

    def forward(self, x):
        skips = []
        x = self.down(x, skip_in=None)
        x = self.residual_0(x, skip_in=None)
        skips.append(x)
        x = self.residual_1(x, skip_in=None)
        skips.append(x)
        return x, skips
    
    def __call__(self, *args, **kwargs):
        return super(DownBlock, self).__call__(*args, **kwargs)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_middle, c_out, up_mode, drop_prob, w_norm):
        super(UpBlock, self).__init__()

        self.up_mode = up_mode
        self.w_norm = w_norm
        self.drop_prob = drop_prob
        
        self.residual_0 = Residual(c_in=c_in, c_out=c_middle, activation=Activation('elu'), drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.residual_1 = Residual(c_in=c_in, c_out=c_middle, activation=Activation('elu'), drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.up = UpSample(c_in=c_middle, c_out=c_out, w_norm=self.w_norm, mode=self.up_mode)

    def forward(self, x, skip_a, skip_b):
        x = self.residual_0(x, skip_in=skip_a)
        x = self.residual_1(x, skip_in=skip_b)
        x = self.up(x, skip_in=None)
        return x

    def __call__(self, *args, **kwargs):
        return super(UpBlock, self).__call__(*args, **kwargs)


class InitBlock(nn.Module):
    def __init__(self, c_in, c_out, drop_prob, w_norm):
        super(InitBlock, self).__init__()

        self.w_norm = w_norm
        self.drop_prob = drop_prob

        self.nin = NiN(c_in=c_in, c_out=c_out, w_norm=self.w_norm)
        self.residual_0 = Residual(c_in=c_out, c_out=c_out, activation=Activation('elu'), drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
        self.residual_1 = Residual(c_in=c_out, c_out=c_out, activation=Activation('elu'), drop_prob=self.drop_prob,
                                   w_norm=self.w_norm)
    
    def forward(self, x):
        skips = []
        x = self.nin(x)
        x = self.residual_0(x, skip_in=None)
        skips.append(x)
        x = self.residual_1(x, skip_in=None)
        skips.append(x)
        return x, skips

    def __call__(self, *args, **kwargs):
        return super(InitBlock, self).__call__(*args, **kwargs)


class EndBlock(nn.Module):
    def __init__(self, c_in, c_middle, c_out, drop_prob, w_norm):
        super(EndBlock, self).__init__()

        self.w_norm = w_norm
        self.drop_prob = drop_prob

        self.residual_0 = Residual(c_in=c_in, c_out=c_middle, activation=Activation('elu'), drop_prob=self.drop_prob,
                                        w_norm=self.w_norm)
        self.residual_1 = Residual(c_in=c_in, c_out=c_middle, activation=Activation('elu'), drop_prob=self.drop_prob,
                                        w_norm=self.w_norm)
        self.conv = MyConv2d(c_in=c_middle, c_out=c_out, kernel_size=3, stride=1, padding=1, w_norm=self.w_norm)

    def forward(self, x, skip_a, skip_b):
        x = self.residual_0(x, skip_in=skip_a)
        x = self.residual_1(x, skip_in=skip_b)
        x = self.conv(x, skip_in=None)
        return x

    def __call__(self, *args, **kwargs):
        return super(EndBlock, self).__call__(*args, **kwargs)


class Vunet_fix_res(nn.Module):
    def __init__(self, args: argparse.Namespace):
        """
        :param args: Command line arguments for current experiment
        """
        super(Vunet_fix_res, self).__init__()

        self.args = args
        # Weight normalization, see: https://arxiv.org/pdf/1602.07868.pdf
        self.w_norm = args.w_norm
        self.drop_prob = args.drop_prob
        self.up_mode = args.up_mode
        self.vunet_256 = args.vunet_256

        """
        Appearance Encoder
        """
        app_c_in = 6
        # 1 block
        self.app_encoder_1 = InitBlock(c_in=app_c_in, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # from 32 to 128
        self.app_encoder_1_a = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)
        self.app_encoder_1_b = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)

        if self.vunet_256:
            self.app_encoder_1_c = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # 2 block
        self.app_encoder_2 = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 3 block
        self.app_encoder_3 = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 4 block
        self.app_encoder_4 = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # Skips
        # 3
        self.app_skip_3_c = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        # 4
        self.app_skip_4_c = NiN(c_in=128, c_out=128, w_norm=self.w_norm)

        """
        Appearance Decoder
        """
        self.app_bottleneck = MyConv2d(c_in=128, c_out=128, kernel_size=1, stride=1, padding=0, w_norm=self.w_norm)
        # 1 block
        self.app_decoder_1_a = Residual(c_in=256, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                        w_norm=self.w_norm)
        self.app_decoder_1_b = Sampler(c_in=128, c_out=128, w_norm=self.w_norm)
        self.app_decoder_1_c = MyConv2d(c_in=256, c_out=128, kernel_size=1, stride=1, padding=0, w_norm=self.w_norm)

        self.app_decoder_1_d = Residual(c_in=256, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                        w_norm=self.w_norm)
        self.app_decoder_1_e = UpSample(c_in=128, c_out=128, w_norm=self.w_norm, mode=self.up_mode)

        # 2 block
        self.app_decoder_2_a = Residual(c_in=128, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                        w_norm=self.w_norm)  # todo `c_in` was 256 when using skip
        self.app_decoder_2_b = Sampler(c_in=128, c_out=128, w_norm=self.w_norm)

        """
        Shape Encoder
        """
        # 1 block
        self.shape_encoder_1 = InitBlock(c_in=3, c_out=32, drop_prob=self.drop_prob, w_norm=self.w_norm)

        if self.vunet_256:
            # 1_a block
            self.shape_encoder_1_a = DownBlock(c_in=32, c_out=32, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # 2 block
        self.shape_encoder_2 = DownBlock(c_in=32, c_out=64, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 3 block
        self.shape_encoder_3 = DownBlock(c_in=64, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 4 block
        self.shape_encoder_4 = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 5 block
        self.shape_encoder_5 = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 6 block
        self.shape_encoder_6 = DownBlock(c_in=128, c_out=128, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # Skips
        # 1
        self.shape_skip_1_b = NiN(c_in=32, c_out=32, w_norm=self.w_norm)
        self.shape_skip_1_c = NiN(c_in=32, c_out=32, w_norm=self.w_norm)

        if self.vunet_256:
            # 1_a
            self.shape_skip_1_a_b = NiN(c_in=32, c_out=32, w_norm=self.w_norm)
            self.shape_skip_1_a_c = NiN(c_in=32, c_out=32, w_norm=self.w_norm)
        # 2
        self.shape_skip_2_b = NiN(c_in=64, c_out=64, w_norm=self.w_norm)
        self.shape_skip_2_c = NiN(c_in=64, c_out=64, w_norm=self.w_norm)
        # 3
        self.shape_skip_3_b = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        self.shape_skip_3_c = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        # 4
        self.shape_skip_4_b = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        self.shape_skip_4_c = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        # 5
        self.shape_skip_5_b = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        self.shape_skip_5_c = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        # 6
        self.shape_skip_6_b = NiN(c_in=128, c_out=128, w_norm=self.w_norm)
        self.shape_skip_6_c = NiN(c_in=128, c_out=128, w_norm=self.w_norm)

        """
        Shape Decoder
        """
        self.shape_bottleneck = MyConv2d(c_in=128, c_out=128, kernel_size=1, stride=1, padding=0, w_norm=self.w_norm)

        # 1 block with auto-regression
        self.shape_decoder_1 = AutoRegressiveBlock(drop_prob=self.drop_prob, w_norm=self.w_norm)

        self.shape_decoder_1_n = NiN(c_in=256, c_out=128, w_norm=self.w_norm)
        self.shape_decoder_1_o = Residual(c_in=256, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                          w_norm=self.w_norm)
        self.shape_decoder_1_p = UpSample(c_in=128, c_out=128, w_norm=self.w_norm, mode=self.up_mode)

        # 2 block with auto-regression
        self.shape_decoder_2 = AutoRegressiveBlock(drop_prob=self.drop_prob, w_norm=self.w_norm)

        self.shape_decoder_2_n = NiN(c_in=256, c_out=128, w_norm=self.w_norm)
        self.shape_decoder_2_o = Residual(c_in=256, c_out=128, activation=Activation('elu'), drop_prob=self.drop_prob,
                                          w_norm=self.w_norm)
        self.shape_decoder_2_p = UpSample(c_in=128, c_out=128, w_norm=self.w_norm, mode=self.up_mode)

        # 3 block
        self.shape_decoder_3 = UpBlock(c_in=256, c_middle=128, c_out=128, up_mode=self.up_mode, drop_prob=self.drop_prob, w_norm=self.w_norm)
        # 4 block
        self.shape_decoder_4 = UpBlock(c_in=256, c_middle=128, c_out=64, up_mode=self.up_mode, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # 5 block
        self.shape_decoder_5 = UpBlock(c_in=128, c_middle=64, c_out=32, up_mode=self.up_mode, drop_prob=self.drop_prob, w_norm=self.w_norm)

        if self.vunet_256:
            # 5_a block
            self.shape_decoder_5_a = UpBlock(c_in=64, c_middle=32, c_out=32, up_mode=self.up_mode, drop_prob=self.drop_prob, w_norm=self.w_norm)

        # 6 block
        self.shape_decoder_6 = EndBlock(c_in=64, c_middle=32, c_out=3, drop_prob=self.drop_prob, w_norm=self.w_norm)

    def forward_enc_up(self, x):
        outputs = []
        skips = []
        x, _ = self.app_encoder_1(x)

        x, _ = self.app_encoder_1_a(x)
        x, _ = self.app_encoder_1_b(x)
        if self.vunet_256:
            x, _ = self.app_encoder_1_c(x)

        x, _ = self.app_encoder_2(x)
        x, _ = self.app_encoder_3(x)
        # keep x as skip
        skips.append(self.app_skip_3_c(x))
        x, skips_layer = self.app_encoder_4(x)
        # keep x as skip
        # keep x and skip before as outputs
        outputs.append(skips_layer[-2])
        outputs.append(x)
        skips.append(self.app_skip_4_c(x))
        return outputs, skips

    def forward_dec_up(self, x):
        outputs = []
        skips = []
        x, skips_layer = self.shape_encoder_1(x)
        skips.append(self.shape_skip_1_b(skips_layer[-2]))
        skips.append(self.shape_skip_1_c(skips_layer[-1]))

        if self.vunet_256:
            x, skips_layer = self.shape_encoder_1_a(x)
            skips.append(self.shape_skip_1_a_b(skips_layer[-2]))
            skips.append(self.shape_skip_1_a_c(skips_layer[-1]))

        x, skips_layer = self.shape_encoder_2(x)
        skips.append(self.shape_skip_2_b(skips_layer[-2]))
        skips.append(self.shape_skip_2_c(skips_layer[-1]))

        x, skips_layer = self.shape_encoder_3(x)
        skips.append(self.shape_skip_3_b(skips_layer[-2]))
        skips.append(self.shape_skip_3_c(skips_layer[-1]))

        x, skips_layer = self.shape_encoder_4(x)
        skips.append(self.shape_skip_4_b(skips_layer[-2]))
        skips.append(self.shape_skip_4_c(skips_layer[-1]))

        x, skips_layer = self.shape_encoder_5(x)
        skips.append(self.shape_skip_5_b(skips_layer[-2]))
        skips.append(self.shape_skip_5_c(skips_layer[-1]))

        x, skips_layer = self.shape_encoder_6(x)
        skips.append(self.shape_skip_6_b(skips_layer[-2]))
        skips.append(self.shape_skip_6_c(skips_layer[-1]))

        outputs.append(x)
        return outputs, skips

    def forward_enc_down(self, enc_up_outputs, skips):
        mu = []
        z = []
        x = self.app_bottleneck(enc_up_outputs[-1])
        x = self.app_decoder_1_a(x, skip_in=skips[-1])
        mu_0, z_0 = self.app_decoder_1_b(x)
        mu.append(mu_0)
        z.append(z_0)
        x_ = self.app_decoder_1_c(torch.cat([enc_up_outputs[-2], z[-1]], 1))
        x = self.app_decoder_1_d(x, skip_in=x_)
        x = self.app_decoder_1_e(x, skip_in=None)

        # x = self.app_decoder_2_a(x, skip_in=skips[-2])
        x = self.app_decoder_2_a(x, skip_in=None)  # there is no more 8x8 skip
        mu_1, z_1 = self.app_decoder_2_b(x)
        mu.append(mu_1)
        z.append(z_1)

        return mu, z

    def forward_dec_down(self, dec_up_outputs, skips, enc_down_mu=()):
        mu = []
        z = []
        x = self.shape_bottleneck(dec_up_outputs[-1])

        # autoregressive
        skip_a = skips.pop()
        skip_b = skips.pop()
        enc_down_mu_a = None if len(enc_down_mu) == 0 else enc_down_mu[0]
        x, mu_0, z_0 = self.shape_decoder_1(x, skip_a, enc_down_mu_a)
        mu.append(mu_0)
        z.append(z_0)
        x = self.shape_decoder_1_n(torch.cat([x, z_0], 1))
        x = self.shape_decoder_1_o(x, skip_in=skip_b)
        x = self.shape_decoder_1_p(x, skip_in=None)

        # autoregressive
        skip_a = skips.pop()
        skip_b = skips.pop()
        enc_down_mu_a = None if len(enc_down_mu) == 0 else enc_down_mu[1]
        x, mu_1, z_1 = self.shape_decoder_2(x, skip_a, enc_down_mu_a)
        mu.append(mu_1)
        z.append(z_1)
        x = self.shape_decoder_2_n(torch.cat([x, z_1], 1))
        x = self.shape_decoder_2_o(x, skip_in=skip_b)
        x = self.shape_decoder_2_p(x, skip_in=None)

        skip_a = skips.pop()
        skip_b = skips.pop()
        x = self.shape_decoder_3(x, skip_a, skip_b)

        skip_a = skips.pop()
        skip_b = skips.pop()
        x = self.shape_decoder_4(x, skip_a, skip_b)

        skip_a = skips.pop()
        skip_b = skips.pop()
        x = self.shape_decoder_5(x, skip_a, skip_b)

        if self.vunet_256:
            skip_a = skips.pop()
            skip_b = skips.pop()
            x = self.shape_decoder_5_a(x, skip_a, skip_b)

        skip_a = skips.pop()
        skip_b = skips.pop()
        x = self.shape_decoder_6(x, skip_a, skip_b)
        assert not skips

        return x, mu, z

    def forward(self, y_tilde, x=None, mean_mode='mean_appearance'):
        if self.vunet_256:
            assert y_tilde.shape[-1] == 256
            if x is not None:
                assert x.shape[-1] == 256
        else:
            assert y_tilde.shape[-1] == 128
            if x is not None:
                assert x.shape[-1] == 128

        assert mean_mode in ['mean_appearance', 'mean_shape']
        if mean_mode == 'mean_appearance':
            output_enc_up, skips_enc_up = self.forward_enc_up(x)
            mu_app, z_app = self.forward_enc_down(output_enc_up, skips_enc_up)
            output_dec_up, skips_dec_up = self.forward_dec_up(y_tilde)
            x_tilde, mu_shape, z_shape = self.forward_dec_down(output_dec_up, skips_dec_up, z_app)
            return x_tilde, mu_app, mu_shape
        else:
            output_dec_up, skips_dec_up = self.forward_dec_up(y_tilde)
            x_tilde, mu_shape, z_shape = self.forward_dec_down(output_dec_up, skips_dec_up)
            return x_tilde

    def __call__(self, *args, **kwargs):
        return super(Vunet_fix_res, self).__call__(*args, **kwargs)


if __name__ == '__main__':
    from argparse import Namespace
    vun = Vunet_fix_res(Namespace(w_norm=False, drop_prob=0.0, up_mode='nearest', vunet_256=False))
    res = vun(torch.randn(2, 3, 128, 128), torch.randn(2, 6, 128, 128))
    print(res[0].shape)