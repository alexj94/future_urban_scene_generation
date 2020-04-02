from pathlib import Path

import torch
import torch.nn as nn
from vunet.model.vgg19_caffe import CaffeVGG19


class PerceptualLayer(nn.Module):
    def __init__(self, vgg_path: Path, vgg_pool: str, upsample=False):
        super(PerceptualLayer, self).__init__()

        self.layers = nn.ModuleList()

        net = CaffeVGG19(weights_path=Path(vgg_path), pool_method=vgg_pool)

        # Output of 2nd convolutional layer of ReLu activated blocks
        layers_relu = ['3', '8', '13', '22', '31']

        layers = []
        for name, layer in net.features.named_children():
            if name in layers_relu:
                layers.append(layer)
                self.layers.append(nn.Sequential(*layers))
                layers = []
            else:
                layers.append(layer)

        # Not trainable. Still, we need gradients to flow (can't be with torch.no_grad())
        for w in self.layers.parameters():
            w.requires_grad = False

        self.upsample = upsample

    def forward(self, x: torch.Tensor):
        x[:, 0, ...] = x[:, 0, ...] - 103.939
        x[:, 1, ...] = x[:, 1, ...] - 116.779
        x[:, 2, ...] = x[:, 2, ...] - 123.68
        if self.upsample:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')

        feats = [x]

        for block in self.layers:
            x = block(x)
            feats.append(x)
        return feats

    def __call__(self, *args, **kwargs):
        return super(PerceptualLayer, self).__call__(*args, **kwargs)
