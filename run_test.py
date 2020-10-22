import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision.models as models
from PyQt5 import QtWidgets

from GUI.app_interface import main_GUI
from edgeconnect.models import EdgeModel
from edgeconnect.models import InpaintingModel
from stacked_hourglass.models import HourglassNet
from utils.cad_utils import load_ply_and_3d_kpoints
from utils.inpaint_utils import load_config
from utils.video_info_utils import parse_calibration_file
from utils.video_info_utils import parse_tracking_files
from vunet.models import Vunet_fix_res
from warp_learn.models import G_Resnet


def load_models(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)

    # Loading inpainting model ==> EDGECONNECT
    config = load_config(args)
    edge_model = EdgeModel(config).to(device)
    edge_model.load()
    edge_model.eval()
    inpaint_model = InpaintingModel(config).to(device)
    inpaint_model.load()
    inpaint_model.eval()

    # Loading MaskRCNN model ==> MaskRCNN ResNet50 FPN
    model_maskrcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model_maskrcnn.to(device)
    model_maskrcnn.eval()

    # Loading cad classifier model ==> VGG19
    print('Loading VGG19 model...')
    model_cad = models.vgg19(pretrained=True)
    model_cad.classifier[6] = torch.nn.Linear(4096, 10)
    for param in model_cad.parameters():
        param.requires_grad = False
    map_location = None
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage
    data = torch.load(str(args.checkpoints_dir / 'cads' / 'model.pth'),
                      map_location=map_location)
    model_cad.load_state_dict(data)
    model_cad.to(device)
    model_cad.eval()

    # Loading keypoints localization model ==> STACKED HOURGLASS
    print('Loading HOURGLASS model...')
    model_kp = HourglassNet(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=12)
    map_location = None
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage
    data = torch.load(str(args.checkpoints_dir / 'kpoints' / 'hourglass.pth'),
                      map_location=map_location)
    model_kp.load_state_dict(data)
    model_kp.to(device)
    model_kp.eval()

    # Loading first novel view synthesis model ==> ICN
    print('Loading ICN model...')
    input_nc = 21
    model_icn = G_Resnet(input_nc).to(device)
    model_icn.load_state_dict(torch.load(args.checkpoints_dir / 'icn' / '256_synth' /
                                         'gnet_00020.pth', map_location=torch.device(device)))
    model_icn.eval()

    # Loading second novel view synthesis model ==> VUNET
    print('Loading VUnet model...')
    model_VUnet = Vunet_fix_res(args=Namespace(up_mode='subpixel', w_norm=True, drop_prob=0.2,
                                               vunet_256=True))
    model_VUnet = model_VUnet.to(args.device)
    model_VUnet.load_state_dict(torch.load(args.checkpoints_dir / 'vunet' / '256' / 'vunet.pth',
                                map_location=torch.device(device)))
    model_VUnet.eval()

    return device, config, model_maskrcnn, edge_model, inpaint_model, model_cad, model_kp, model_icn, model_VUnet


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('video_dir', type=Path)
    parser.add_argument('kpoints_dir', type=Path)
    parser.add_argument('checkpoints_dir', type=Path)
    parser.add_argument('--scale_calib', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--det_mode', type=str, default='ssd512',
                        help='Possible values are "yolo3", "ssd512" or "mask_rcnn"')
    parser.add_argument('--track_mode', type=str, default='tc',
                        help='Possible values are "deepsort", "tc" or "moana"')
    parser.add_argument('--bbox_scale', type=float, default=1.)
    parser.add_argument('--video_fps', type=int, default=10)
    # INPAINTING arguments
    parser.add_argument('--inpaint', action='store_true')
    parser.add_argument('--inpaint_model', type=int, choices=[1, 2, 3, 4], default=4,
                        help='1: edge model, 2: inpaint model, 3: edge-inpaint model, '
                             '4: joint model')
    # CAD/KEYPOINTS arguments
    parser.add_argument('--reso', dest='reso', type=int, default=256,
                        help='Insert input resolution of the network')
    parser.add_argument('--batch', dest='batch', type=int, default=1,
                        help='Insert batch size for inference')
    parser.add_argument('--blocks', dest='blocks', type=int, default=1,
                        help='Insert number of blocks in the network')
    parser.add_argument('--stacks', dest='stacks', type=int, default=2,
                        help='Insert number of stacks in the network')
    parser.add_argument('--device', dest='device', default='cuda',
                        help='Insert device for model inference')

    args = parser.parse_args()

    # Get video information
    video_file = str(args.video_dir / 'vdo.avi')
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f'Error opening video "{video_file}"')

    trajectories = parse_tracking_files(video_dir=args.video_dir, track_type=args.track_mode,
                                        det_mode=args.det_mode)

    calib_file = args.video_dir / 'calibration.txt'
    homography_matrix = parse_calibration_file(calib_file)
    _, inv_homography_matrix = cv2.invert(homography_matrix)
    if args.scale_calib:
        scale = 1280 / 1920
        scale_matrix = np.array([[scale, 0, 0],
                                [0, scale, 0],
                                [0, 0, scale]], dtype=np.float64)
        inv_homography_matrix = scale_matrix @ inv_homography_matrix

    # Load networks models
    device, config, maskrcnn_model, edge_model, inpaint_model, model_cad, model_kp, model_icn, model_VUnet = load_models(args)

    cads_ply = []
    kpoints_dicts = []
    _____SCALE_F = 5  # tunable value -- we choose to set all CAD vehicles length to 5 meters
    for i in range(10):
        ply, kpoints_3d_dict = load_ply_and_3d_kpoints(args.kpoints_dir, cad_idx=i)
        ply.vertices = o3d.utility.Vector3dVector(np.asarray(ply.vertices) * _____SCALE_F)
        cads_ply.append(ply)
        kpoints_dicts.append(kpoints_3d_dict)

    # Open app window
    app = QtWidgets.QApplication(sys.argv)
    ex = main_GUI('Future scene synthesis', cap, args.video_dir, trajectories,
                  args.bbox_scale, inv_homography_matrix, args, device, config, maskrcnn_model,
                  edge_model, inpaint_model, model_cad, model_kp, model_icn, model_VUnet,
                  cads_ply, kpoints_dicts, args.inpaint)
    sys.exit(app.exec_())
