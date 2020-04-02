from pathlib import Path
import argparse
from typing import List

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils.normalization import to_image
from datasets.dataset_stick import StickDataset
from vunet.model.vunet_fixed import Vunet_fix_res


def transfer_pass(vunet: Vunet_fix_res,
                  appearance_enc_in: torch.Tensor,  # Input for appearance encoder
                  dest_shapes: List[torch.Tensor],  # Targets shape(s)
                  ):

    output_enc_up, skips_enc_up = vunet.forward_enc_up(appearance_enc_in)
    mu_app, z_app = vunet.forward_enc_down(output_enc_up, skips_enc_up)

    outputs = []
    for target_shape in dest_shapes:
        output_dec_up, skips_dec_up = vunet.forward_dec_up(target_shape.unsqueeze(0))
        x_tilde, mu_shape, z_shape = vunet.forward_dec_down(output_dec_up, skips_dec_up, mu_app)
        outputs.append(x_tilde)
    return torch.cat(outputs, dim=0)


if __name__ == '__main__':
    args = argparse.Namespace(dataset_dir='', demo=..., w_norm=True, drop_prob=0.1, up_mode='subpixel')

    net = Vunet_fix_res(args)
    net = net.to('cuda')
    net.eval()

    dataset_dir = Path('/home/luca/Desktop/vunet-pytorch/data/data/pascal3d_vunet_NBG')

    output_dir = Path('./transfer_output_NBG')
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    side = 128
    dataset = StickDataset(folder=dataset_dir, resize_factor=0.5, demo_mode=True)
    dataset.eval()

    dl_test = DataLoader(dataset, batch_size=8 + 1, shuffle=False, drop_last=True, num_workers=4)

    # Load pre-trained weights
    w_path = '/tmp/2018-12-11_19-24-10/ckpt/00180.pth'
    net.load_state_dict(torch.load(w_path))

    # For each batch, the appearance is conditioned on the first example
    #  and then transferred on all others
    assert dataset.mode == 'eval'
    for i, data in enumerate(dl_test):
        x_input = data['image'][0:1].to('cuda')
        y_input = data['shape_original'][0:1].to('cuda')

        y_targets = data['shape_original'][0:].to('cuda')
        outputs = transfer_pass(vunet=net, appearance_enc_in=x_input, dest_shapes=y_targets)

        sticks_and_x_in = torch.cat([data['shape_original'][0:1], data['image_original'][0:1]], 2)

        sticks_and_x_out = torch.cat([y_targets, outputs], dim=2).to('cpu')

        image = torch.cat([sticks_and_x_in, sticks_and_x_out], dim=0)
        cv2.imwrite(f'{output_dir}/{i:06d}.png', to_image(make_grid(image), from_LAB=True))
