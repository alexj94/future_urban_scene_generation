import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.feature import canny
from tqdm import tqdm

from edgeconnect.config import Config
from edgeconnect.models import EdgeModel, InpaintingModel


def test(args, config):
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda:0")
    else:
        config.DEVICE = torch.device("cpu")

    if not args.output.is_dir():
        args.output.mkdir(exist_ok=True, parents=True)

    cv2.setNumThreads(0)

    # Model loading
    edge_model = EdgeModel(config).to(config.DEVICE)
    inpaint_model = InpaintingModel(config).to(config.DEVICE)
    edge_model.load()
    inpaint_model.load()
    edge_model.eval()
    inpaint_model.eval()

    # Inference on images
    img_list = sorted(args.input.glob('*.png'))
    mask_list = sorted(args.mask.glob('*.png'))
    for i in tqdm(range(len(img_list))):
        img = cv2.imread(str(img_list[i]))
        img = cv2.resize(img, (256, 256))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(str(mask_list[i]))
        mask = cv2.resize(mask, (256, 256))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255
        mask_edge = (1 - mask / 255).astype(np.bool)
        edge = canny(img_gray, config.SIGMA, mask_edge)

        img = transforms.ToTensor()(img).cuda().unsqueeze(0).float()
        img_gray = transforms.ToTensor()(img_gray).cuda().unsqueeze(0).float()
        mask = transforms.ToTensor()(mask).cuda().unsqueeze(0).float()
        edge = transforms.ToTensor()(edge).cuda().unsqueeze(0).float()

        # inference
        with torch.no_grad():
            edge = edge_model(img_gray, edge, mask).detach()
            inpaint = inpaint_model(img, edge, mask)
        output_merged = (inpaint * mask) + (img * (1 - mask))

        # postprocess inpainted image
        output = output_merged * 255.0
        output = output.permute(0, 2, 3, 1)
        output = output.squeeze(0).cpu().numpy().astype(np.uint8)

        cv2.imwrite(str(args.output / f'{img_list[i].name}'), output)


def load_config(args):
    """
        Loads model config
    """
    config_path = os.path.join(args.path, 'config.yml')
    config = Config(config_path)
    config.MODE = 2
    config.MODEL = args.model if args.model is not None else 3
    config.INPUT_SIZE = 0
    if args.input is not None:
        config.TEST_FLIST = args.input
    if args.mask is not None:
        config.TEST_MASK_FLIST = args.mask
    if args.edge is not None:
        config.TEST_EDGE_FLIST = args.edge
    if args.output is not None:
        config.RESULTS = args.output

    return config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4],
                        help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
    parser.add_argument('--input', type=Path, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=Path, help='path to the masks directory or a mask file')
    parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
    parser.add_argument('--output', type=Path, help='path to the output directory')

    args = parser.parse_args()

    config = load_config(args)
    test(args, config)
