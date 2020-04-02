# -*- coding: utf-8 -*-
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from integration.online_visibility import pascal_texture_planes
from integration.planes_utils import to_image
from integration.planes_utils import warp_unwarp_planes
from integration.vehicle_utils import get_central_crop
from integration.vehicle_utils import get_vehicle_information
from integration.von import G_Resnet
from integration.von import get_icn_inputs
from utils.cad_utils import load_ply_and_3d_kpoints
from utils.geometry import extrinsic_from_rodrigues
from utils.geometry import z_rot
from utils.keypoint_utils import kpoints_array_to_dict
from utils.keypoint_utils import kpoints_dict_to_array


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('debug_data_dir', type=Path,
                        help='Directory containing debug data')
    parser.add_argument('model_path', type=Path,
                        help='Path to pre-trained model')
    parser.add_argument('cad_root', type=Path,
                        help='Directory containing 3D CAD')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    for d in [args.debug_data_dir, args.cad_root]:
        if not d.is_dir():
            raise FileNotFoundError(f'{d} not valid.')

    if not args.model_path.is_file():
        raise FileNotFoundError(f'{args.model_path} not valid.')

    input_nc = 21
    icn_w, icn_h = 256, 256

    icn = G_Resnet(input_nc).to(args.device)
    icn.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
    icn.eval()

    frame_path = args.debug_data_dir / 'frame.png'
    frame = cv2.imread(str(frame_path))

    npz_path = args.debug_data_dir / 'data_generation.npz'
    npz_data = np.load(npz_path)

    bbox = npz_data['bbox']

    ply, kpoints_3d_dict = load_ply_and_3d_kpoints(args.cad_root, cad_idx=npz_data['cad_idx'])
    rvect = npz_data['rvect']
    tvect = npz_data['tvect']
    intrinsic = npz_data['K']
    extrinsic = extrinsic_from_rodrigues(rvect, tvect)
    kpoints_2d = npz_data['keypoints_pred'].squeeze()

    # ================ START ===========
    # --------- SRC
    # get central crop (app. prior)
    central_crop = get_central_crop(bbox, frame, icn_w, icn_h)
    # get planes and stuff
    src_sketch_normal, src_sketch_mask, src_planes, src_planes_kpoints, src_planes_visibilities = get_vehicle_information(ply, frame, extrinsic, intrinsic, kpoints_array_to_dict(kpoints_2d), kpoints_3d_dict)

    # --------- DST
    # do a transformation and get new 2D keypoints
    theta = np.radians(0)
    delta_t = [0., -1.8, 0.]
    delta_rot = z_rot(theta)
    for k, v in kpoints_3d_dict.items():
        kpoints_3d_dict[k] = v @ delta_rot + delta_t
    ply.vertices = o3d.Vector3dVector(np.asarray(ply.vertices) @ delta_rot + delta_t)
    kpoints_2d_next, _ = cv2.projectPoints(kpoints_dict_to_array(kpoints_3d_dict, dim=3), rvect,
                                           tvect, intrinsic, np.zeros((1, 5)))
    kpoints_2d_next = kpoints_2d_next.squeeze(1)
    # get planes and stuff
    dst_sketch_normal, dst_sketch_mask, dst_planes, dst_planes_kpoints, dst_planes_visibilities = get_vehicle_information(
        ply, frame, extrinsic, intrinsic, kpoints_array_to_dict(kpoints_2d_next), kpoints_3d_dict)
    # ---------- WARP, FORWARD  AND STITCH
    # WARP
    planes_warped, _ = warp_unwarp_planes(src_planes, src_planes_kpoints, dst_planes_kpoints,
                                          src_planes_visibilities, dst_planes_visibilities, 'car',
                                          pascal_texture_planes)
    # INVERT MASK-> True where the vehicle is
    dst_sketch_mask = np.logical_not(dst_sketch_mask)

    # GET CROPPED INPUTS
    icn_input, crop_info = get_icn_inputs(planes_warped, dst_sketch_normal, dst_sketch_mask, central_crop, icn_w, icn_h)
    # FORWARD
    net_image = to_image(icn(icn_input.to(args.device))[0], from_LAB=True)

    # REVERT AND STITCH
    crop_size_orig = crop_info['crop_size_orig']
    pad_xy_before = crop_info['pad_xy_before']
    pad_xy_after = crop_info['pad_xy_after']
    crop_xy_min = crop_info['crop_xy_min']

    crop_inv = cv2.resize(net_image, crop_size_orig[::-1])
    crop_inv = crop_inv[pad_xy_before[1]:crop_inv.shape[0] - pad_xy_after[1],
               pad_xy_before[0]:crop_inv.shape[1] - pad_xy_after[0]]  # revert padding

    out_frame = np.zeros_like(frame)
    out_frame[crop_xy_min[1]: crop_xy_min[1] + crop_size_orig[0],
    crop_xy_min[0]: crop_xy_min[0] + crop_size_orig[1]] = crop_inv
    frame[dst_sketch_mask] = out_frame[dst_sketch_mask]

    for i, plane in enumerate(planes_warped):
        cv2.imshow(f'plane_{i}', plane)

    cv2.imshow('res', net_image)
    cv2.imshow('src_central_crop', central_crop)
    cv2.imshow('dst_normal', dst_sketch_normal)
    cv2.imshow('src_normal', src_sketch_normal)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)




