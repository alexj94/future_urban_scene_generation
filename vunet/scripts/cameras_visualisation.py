# TODO MOVE THIS SCRIPT, IT DOESN'T DEPENDS ON VUNET
import argparse
import collections
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from open3d import Vector3dVector
from open3d import Vector3iVector
from open3d import draw_geometries_with_key_callbacks
from open3d import read_triangle_mesh
from tqdm import tqdm

from utils.geometry import y_rot
from utils.geometry import z_rot
from datasets.dataset_stick import StickDataset
from utils.normalization import to_image

Kpoint3D = Sequence[float]
global geometries
global triangles_base
global cam_index
global dd
global colors


def get_cameras(camera_mesh, angle_y_array, angle_z_array, radius_array, synth=True, color=(0, 0, 0)):
    assert len(angle_y_array) == len(angle_z_array) == len(radius_array)
    camera_mesh = deepcopy(camera_mesh)
    N = len(angle_y_array)
    vertices_base = np.asarray(camera_mesh.vertices).copy()
    triangles_base = np.asarray(camera_mesh.triangles).copy()
    # prepare buffers to be filled with translated meshes
    vertices = np.zeros((N, *vertices_base.shape), dtype=vertices_base.dtype)
    triangles = np.zeros((N, *triangles_base.shape), dtype=triangles_base.dtype)
    
    for i in tqdm(range(N)):
        # ------------------ Matrix computation -------------
        angle_y = angle_y_array[i]
        angle_z = angle_z_array[i]
        radius = radius_array[i]
        if not synth:
            angle_y = 90 - angle_y
            angle_z = angle_z - 90

        # TODO check ranges
        # TODO replace this using code from geoemtry
        rot_m = z_rot(alpha=float(np.radians(angle_z))) @ y_rot(alpha=float(np.radians(angle_y)))
        rot_m[:, 0], rot_m[:, 1] = rot_m[:, 1].copy(), rot_m[:, 0].copy()
        # translate axis using new z direction
        translation = rot_m[:, -1].copy()
        translation /= np.linalg.norm(translation)
        translation *= radius
        # mirror z by negating last cols ([0,0,1] projection)
        rot_m[:, -1] *= -1
        rot_m = np.concatenate([rot_m, translation[..., None]], -1)
        # ------------------- Box update -----------------------------
        new_triangles = triangles_base + (np.max(triangles_base) + 1) * i
        new_vertices = np.concatenate([vertices_base, np.ones((len(vertices_base), 1))], -1)
        vertices[i] = (rot_m @ new_vertices.T).T
        triangles[i] = new_triangles

    camera_mesh.vertices = Vector3dVector(vertices.reshape(-1, 3))
    camera_mesh.triangles = Vector3iVector(triangles.reshape(-1, 3))
    camera_mesh.compute_vertex_normals()
    camera_mesh.paint_uniform_color(color)

    return camera_mesh, triangles_base, vertices_base


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
        global cam_index
        # zero case is idle
        if self.key == 0:
            pass
        if self.key == ord('N'):
            geometries['mesh_synth'].paint_uniform_color(colors['mesh_synth'])
            colors_mesh = np.asarray(geometries['mesh_synth'].vertex_colors)
            index_to_color = triangles_base + (np.max(triangles_base) + 1) * cam_index
            colors_mesh[index_to_color.flatten()] = np.asarray([1, 0, 0])[None, :]
            geometries['mesh_synth'].vertex_colors = Vector3dVector(colors_mesh)
            cv2.imshow('w', to_image(dd[cam_index], from_LAB=True))
            cv2.waitKey(30)
            cam_index += 1
            pass

        vis.update_geometry()


def main():
    global geometries
    global triangles_base
    global cam_index
    global dd
    global colors
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir_1', type=Path, help='Dataset 1 directory')
    parser.add_argument('dataset_dir_2', type=Path, help='Dataset 2 directory')
    parser.add_argument('camera_ply_path', type=Path, help='path to the object placed as camera')
    parser.add_argument('car_ply_path', type=Path, help='path to the object placed in the origin')
    parser.add_argument('--dataset_1_is_synth', action='store_true')
    parser.add_argument('--dataset_2_is_synth', action='store_true')
    args = parser.parse_args()

    dd = {}
    colors = {}
    cam_index = 0
    geometries = Geometries()
    # --------------------- Mesh --------------------------------
    mesh = read_triangle_mesh(str(args.camera_ply_path))
    mesh.vertices = Vector3dVector(np.asarray(mesh.vertices) * 0.1)  # todo fix in  blender
    # --------------------- REAL -------------------------------------
    # --------------------- Dataset ----------------------------
    dataset = StickDataset(folder=args.dataset_dir_1, resize_factor=0.5,
                           demo_mode=True)
    dataset.eval()
    N = len(dataset)
    angle_y_array = np.zeros(N)
    angle_z_array = np.zeros(N)
    radius_array = np.zeros(N)
    errors_array = np.zeros(N)
    for i in tqdm(range(N)):
        # ------------------ Store Parameters -------------
        vpoint = dataset[i]['image_meta']['vpoint']
        angle_y_array[i] = vpoint[1]
        angle_z_array[i] = vpoint[0]
        radius_array[i] = vpoint[3]
        errors_array[i] = vpoint[-4]

        dd[i] = dataset[i]['image_log']

    camera_mesh, triangles_base, vertices_base = get_cameras(mesh, angle_y_array, angle_z_array, radius_array,
                                                             synth=args.dataset_1_is_synth, color=(0, 1, 0))
    geometries['mesh_synth'] = camera_mesh
    colors['mesh_synth'] = (0, 1, 0)
    # --------------------- SYNTH -------------------------------------
    # --------------------- Dataset ----------------------------
    dataset = StickDataset(folder=args.dataset_dir_2, resize_factor=0.5,
                           demo_mode=True)
    dataset.eval()
    N = len(dataset)
    angle_y_array = np.zeros(N)
    angle_z_array = np.zeros(N)
    radius_array = np.zeros(N)

    for i in tqdm(range(N)):
        # ------------------ Store Parameters -------------
        angle_y_array[i] = dataset[i]['image_meta']['elevation']
        angle_z_array[i] = dataset[i]['image_meta']['azimuth']
        radius_array[i] = dataset[i]['image_meta']['radius']

    camera_mesh, triangles_base, vertices_base = get_cameras(mesh, angle_y_array, angle_z_array, radius_array,
                                                             synth=args.dataset_2_is_synth, color=(0, 0, 1))
    geometries['mesh_real'] = camera_mesh
    colors['mesh_real'] = (0, 0, 1)
    # ------------------------ Car Mesh ------------------------------------
    car = read_triangle_mesh(str(args.car_ply_path))
    car.compute_vertex_normals()
    geometries['car'] = car

    key_callbacks = {
        ord('N'): Callbacks(ord('N'))
    }
    draw_geometries_with_key_callbacks(geometries.as_list(), key_callbacks,
                                       width=1920 // 2, height=1080 // 2, left=50, top=1080//4)


if __name__ == '__main__':
    main()
