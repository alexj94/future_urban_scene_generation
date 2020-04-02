
import argparse
import collections
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from open3d import Vector3dVector
from open3d import Visualizer
from open3d import create_mesh_coordinate_frame
from open3d import draw_geometries_with_key_callbacks
from open3d import read_triangle_mesh
from torchvision.utils import make_grid

from datasets.interop import pascal_idx_to_kpoint
from datasets.interop import pascal_stick_planes
from preprocessing.draw_sticks import draw_sticks
from utils.geometry import intrinsic_matrix
from utils.geometry import pascal_vpoint_to_extrinsics
from utils.geometry import project_points
from utils.normalization import to_image
from utils.normalization import to_tensor
from utils.open3d import draw_segments
from datasets.dataset_stick import StickDataset
from vunet.model.vunet_fixed import Vunet_fix_res as Vunet_128

global camera_vertices_initial
global angle_y
global angle_z
global radius
global kpoints_3d
global geometries
global net
global dataset
global dataset_index

global posterior_vunet
global input_data
global model_idx
global car_cad_kpoints_3D

global focal
global args


Kpoint3D = Sequence[float]


class Geometries(dict):
    def __init__(self):
        super(Geometries, self).__init__()

    def as_list(self):
        l = []
        for v in self.values():
            if isinstance(v, collections.Iterable):
                l.extend(v)
            else:
                l.append(v)
        return l


class Callbacks(object):
    def __init__(self, key: int):
        self.key = key

    def __call__(self, vis):
        global angle_y, angle_z, radius, geometries, posterior_vunet, input_data, dataset_index, model_idx, kpoints_3d, focal
        # zero case is idle
        if self.key == 0:
            pass
        # R means reset
        elif self.key == ord('R'):
            angle_y = 0.
            angle_z = 0.
            radius = 0.
        # inc focal
        elif self.key == ord('O'):
            focal += 100
        # dec focal
        elif self.key == ord('P'):
            focal -= 100
        # rotation along Y
        elif self.key == ord('F'):
            angle_y += 5
        elif self.key == ord('D'):
            angle_y -= 5
        # rotation along z
        elif self.key == ord('S'):
            angle_z += 5
        elif self.key == ord('A'):
            angle_z -= 5
        # displace radius
        elif self.key == ord('H'):
            radius += 0.05
        elif self.key == ord('G'):
            radius -= 0.05
        elif self.key == ord(' '):
            input_data = dataset[dataset_index]
            posterior_vunet = get_style(net, input_data['app_input'].unsqueeze(0).to('cuda'))
            dataset_index += 1

        elif self.key == ord('N'):
            model_idx += 1
            if model_idx == 10:
                model_idx = 0
            car_sticks = stick_line_sets(car_cad_kpoints_3D['kpoints'][model_idx].astype(np.float32))
            geometries['car_sticks_original'] = car_sticks
            kpoints_3d = car_cad_kpoints_3D['kpoints'][model_idx].astype(np.float32) * 0.25   # to reduce camera distances
        else:
            raise NotImplementedError()
        # ensure ranges
        angle_y = np.clip(angle_y, -90, 90)
        angle_z %= 360
        radius = np.clip(radius, 0, radius)

        img_h, img_w = 256, 256
        print('Focal', focal)
        intrinsic = intrinsic_matrix(focal, cx=img_w/2, cy=img_h/2)

        extrinsic = pascal_vpoint_to_extrinsics(az_deg=angle_z + 90,
                                                el_deg=90 - angle_y,
                                                radius=radius)

        # Move the camera mesh
        camera_vertices_step = np.concatenate([camera_vertices_initial, np.ones((len(camera_vertices_initial), 1))], -1)
        camera_rt = np.linalg.inv(np.concatenate([extrinsic, np.asarray([0, 0, 0, 1])[None, ...]], 0))[:-1]
        geometries['camera'].vertices = Vector3dVector((camera_rt @ camera_vertices_step.T).T)

        # Project model kpoints in 2D
        kpoints_2d_step = project_points(kpoints_3d, intrinsic, extrinsic)

        # Mark as invalid points outside the image
        kpoints_2d_step[kpoints_2d_step[:, 0] > img_w, 0] *= -1
        kpoints_2d_step[kpoints_2d_step[:, 1] > img_h, 1] *= -1
        kpoints_2d_step /= (img_w, img_h)
        kpoints_2d_step = np.clip(kpoints_2d_step, -1, 1)
        kpoints_2d_step_dict = {pascal_idx_to_kpoint['car'][i]: kpoints_2d_step[i] for i in range(kpoints_2d_step.shape[0])}
        stick_image = draw_sticks('car', (img_h, img_w), kpoints_2d_step_dict)

        show_size = stick_image.shape[:2][::-1]
        image_to_show = stick_image

        try:
            # transfer_in = input_data['shape_input'].unsqueeze(0).to('cuda')
            stick_image = cv2.resize(stick_image, (128, 128))
            stick_image = cv2.cvtColor(stick_image, cv2.COLOR_BGR2LAB)
            transfer_in = to_tensor(stick_image).unsqueeze(0).to('cuda')

            shaded_show = cv2.resize(to_image(transfer_in.squeeze(0), from_LAB=True), dsize=show_size)
            output_dec_up, skips_dec_up = net.forward_dec_up(transfer_in)
            x_tilde, _, _ = net.forward_dec_down(output_dec_up, skips_dec_up, posterior_vunet)

            x_tilde = to_image(make_grid(x_tilde), from_LAB=True)
            x_tilde = cv2.resize(x_tilde, dsize=show_size)

            transfer_in1 = input_data['shape_input'].unsqueeze(0).to('cuda')

            output_dec_up1, skips_dec_up1 = net.forward_dec_up(transfer_in1)
            x_tilde1, _, _ = net.forward_dec_down(output_dec_up1, skips_dec_up1, posterior_vunet)
            x_tilde1 = to_image(make_grid(x_tilde1), from_LAB=True)
            x_tilde1 = cv2.resize(x_tilde1, dsize=show_size)

            image_original = to_image(input_data['image_log'], from_LAB=True)
            image_original = cv2.resize(image_original, dsize=show_size)
            _ = cv2.resize(to_image(input_data['shape_input'], from_LAB=True), show_size)
            image_to_show = np.concatenate([image_to_show,
                                            shaded_show,
                                            image_original,
                                            x_tilde,
                                            _,
                                            x_tilde1], axis=1)
        except NameError:
            pass
        cv2.imshow('Projection', image_to_show)
        cv2.waitKey(30)

        vis.update_geometry()


def stick_line_sets(kpoint_array: np.ndarray):
    kpoints_3d_dict = {
        pascal_idx_to_kpoint['car'][i]: kpoint_array[i] for i in range(kpoint_array.shape[0])
    }

    l1 = draw_segments(pascal_stick_planes['car']['left'], kpoints_3d_dict, color=(0, 255, 0))
    l2 = draw_segments(pascal_stick_planes['car']['right'], kpoints_3d_dict, color=(255, 0, 0))
    l3 = draw_segments(pascal_stick_planes['car']['wheel'], kpoints_3d_dict, color=(0, 0, 255))
    l4 = draw_segments(pascal_stick_planes['car']['light'], kpoints_3d_dict, color=(0, 0, 255))
    l5 = draw_segments(pascal_stick_planes['car']['roof'], kpoints_3d_dict, color=(0, 0, 255))
    return [l1, l2, l3, l4, l5]


def get_style(net, input_image):
    output_enc_up, skips_enc_up = net.forward_enc_up(input_image)
    mu_app, z_app = net.forward_enc_down(output_enc_up, skips_enc_up)
    return mu_app


def main():
    global camera_vertices_initial
    global angle_y
    global angle_z
    global radius
    global kpoints_3d
    global geometries
    global net
    global dataset
    global dataset_index
    global model_idx
    global posterior_vunet
    global input_data
    global car_cad_kpoints_3D
    global focal
    global args

    geometries = Geometries()

    # Load 3D CADs
    pascal3d_cads_file = f'data/car_cads_swap_1.npz'

    model_idx = 0
    focal = 750

    origin = create_mesh_coordinate_frame(size=0.35, origin=[0, 0, 0])
    geometries['origin'] = origin

    camera = read_triangle_mesh('data/camera.ply')
    camera.vertices = Vector3dVector(np.asarray(camera.vertices) * 0.1)
    camera.compute_vertex_normals()

    # camera = create_mesh_coordinate_frame(size=0.15, origin=[0, 0, 0])
    geometries['camera'] = camera

    angle_y, angle_z, radius = 75., 0., 1.0
    camera_vertices_initial = np.asarray(camera.vertices).copy()

    args = argparse.Namespace(dataset_dir='', demo=..., w_norm=True, drop_prob=0.1,
                              up_mode='subpixel', device='cuda', shaded=False)
    net = Vunet_fix_res(args)
    net = net.to(args.device)

    # Load pre-trained weights
    w_path = '/tmp/2019-01-19_17-25-41/ckpt/00040.pth'  # avg pool, gram
    net.load_state_dict(torch.load(w_path))
    net.eval()

    dataset_dir = Path('/home/luca/Desktop/datasets/pascal_car_stick')

    dataset = StickDataset(folder=dataset_dir, resize_factor=0.5, demo_mode=True)
    dataset.eval()
    dataset_index = 0

    # camera.vertices = Vector3dVector(np.asarray(mesh_frame.vertices) + 3)
    key_callbacks = {
        ord('F'): Callbacks(ord('F')),
        ord('D'): Callbacks(ord('D')),
        ord('A'): Callbacks(ord('A')),
        ord('S'): Callbacks(ord('S')),
        ord('H'): Callbacks(ord('H')),
        ord('G'): Callbacks(ord('G')),
        ord('R'): Callbacks(ord('R')),
        ord(' '): Callbacks(ord(' ')),
        ord('N'): Callbacks(ord('N')),
        ord('O'): Callbacks(ord('O')),
        ord('P'): Callbacks(ord('P')),
    }

    # call callback to update matrix
    Callbacks(ord('N'))(Visualizer())  # init model
    Callbacks(ord(' '))(Visualizer())  # init appearance
    Callbacks(0)(Visualizer())
    draw_geometries_with_key_callbacks(geometries.as_list(), key_callbacks,
                                       width=1920 // 2, height=1080 // 2, left=50, top=1080//4)
    cv2.namedWindow('Projection')


if __name__ == '__main__':
    main()
