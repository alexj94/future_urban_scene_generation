# -*- coding: utf-8 -*-
"""
Online visibility model
"""
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np

from utils.misc_utils import Color

pascal_texture_planes = {
    'car': {
        'left': ['left_back_trunk', 'left_back_wheel', 'left_front_wheel',
                 'left_front_light', 'upper_left_windshield',
                 'upper_left_rearwindow'],
        'right': ['right_back_trunk', 'right_back_wheel', 'right_front_wheel',
                  'right_front_light', 'upper_right_windshield',
                  'upper_right_rearwindow'],
        'roof': ['upper_left_rearwindow', 'upper_left_windshield',
                 'upper_right_windshield', 'upper_right_rearwindow'],
        'front': ['left_front_light', 'right_front_light',
                  'upper_right_windshield', 'upper_left_windshield'],
        'back': ['left_back_trunk', 'right_back_trunk',
                 'upper_right_rearwindow', 'upper_left_rearwindow']
    },
    'chair': {}
}


def project_points(points_3d: np.array,
                   intrinsic: np.array,
                   extrinsic: np.array
                   ) -> np.array:
    """
    Project 3D points in 2D according to pinhole camera model.

    :param points_3d: 3D points to be projected (n_points, 3) 
    :param intrinsic: Intrinsics camera matrix
    :param extrinsic: Extrinsics camera matrix
    :return projected: 2D projected points (n_points, 2) 
    """
    n_points = points_3d.shape[0]

    assert points_3d.shape == (n_points, 3)
    assert extrinsic.shape == (3, 4) or extrinsic.shape == (4, 4)
    assert intrinsic.shape == (3, 3)

    if extrinsic.shape == (4, 4):
        if not np.all(extrinsic[-1, :] == np.asarray([0, 0, 0, 1])):
            raise ValueError('Format for extrinsic not valid')
        extrinsic = extrinsic[:3, :]

    points3d_h = np.concatenate([points_3d, np.ones(shape=(n_points, 1))], 1)

    projected = intrinsic @ extrinsic @ points3d_h.T
    projected /= projected[2, :]
    projected = projected.T
    return projected[:, :2]


def camera_planes_dist(extrinsic, planes3d):
    # Compute the 3d distance of each plane from the camera
    camera_roto = np.linalg.inv(extrinsic)
    cam_x, cam_y, cam_z = camera_roto[:3, -1]

    camera_pos = np.stack([cam_x, cam_y, cam_z])[np.newaxis, ...]

    dist_dict = defaultdict(float)
    mean_kpoints = {}
    for plane_name, plane_kpoints in planes3d.items():
        mean_kp = np.mean(plane_kpoints, 0)
        d = np.linalg.norm(camera_pos - mean_kp, axis=1)
        dist_dict[plane_name] = d

        mean_kpoints[plane_name] = mean_kp

    return dist_dict, mean_kpoints


def draw_plane_simple(pl_name, kpoints2d_dict, color, h, w, pascal_texture_planes, pl_image=None):
    kp_names = pascal_texture_planes['car'][pl_name]  # get kpoints of the plane
    vertices = [tuple(map(int, kpoints2d_dict[name])) for name in kp_names]  # get vertices of keypoints

    if pl_image is None:
        pl_image = np.zeros((h, w, 3), dtype=np.uint8)
    pl_image = cv2.fillPoly(pl_image, [np.asarray(vertices).reshape((-1, 1, 2))],
                            color)
    return pl_image


def draw_plane_occlusion(pl_name, pl_names, distances, kpoints2d_dict, h, w, pascal_texture_planes):
    # draw the plane but remove the ones closer to the camera
    pl_image = draw_plane_simple(pl_name, kpoints2d_dict, Color.WHITE, h, w, pascal_texture_planes)

    for p in pl_names:
        if distances[p] < distances[pl_name]:
            pl_image = draw_plane_simple(p, kpoints2d_dict, Color.BLACK, h, w,
                                         pascal_texture_planes, pl_image)
    return pl_image


def get_plane_area(pl_image: np.ndarray):
    assert len(pl_image.shape) == 3
    return np.sum(pl_image[..., 0] > 0)


def compute_visibility(extrinsic, intrinsic, kpoints_3d, h, w):

    # The following two planes are added just for the sake of correct behavior of
    #  this function. Indeed, although we do not want to add them to warped planes in
    #  deployment, we must add them here to handle occlusion correctly.
    pascal_texture_planes_extended = deepcopy(pascal_texture_planes)
    pascal_texture_planes_extended['car']['front_bt'] = ['left_front_light', 'right_front_light',
                                                         'right_front_wheel', 'left_front_wheel']
    pascal_texture_planes_extended['car']['back_bt'] = ['left_back_trunk', 'right_back_trunk',
                                                        'right_back_wheel', 'left_back_wheel']

    planes_3d = defaultdict(list)
    for plane_name in pascal_texture_planes_extended['car'].keys():
        pl_kp_names = pascal_texture_planes_extended['car'][plane_name]
        for kp_name in pl_kp_names:
            planes_3d[plane_name].append(kpoints_3d[kp_name])

    # Compute the distance of each 3d plane from the camera
    distances_3d, mean_kpoints3d = camera_planes_dist(extrinsic, planes_3d)

    # 3D 2D
    kpoints_2d = {}
    for k_name, kpoint_3d in kpoints_3d.items():
        kpoint_2d = project_points(points_3d=np.asarray(kpoint_3d)[np.newaxis],
                                   intrinsic=intrinsic,
                                   extrinsic=extrinsic)
        kpoints_2d[k_name] = kpoint_2d.squeeze(0)

    # Compute areas
    # Draw the image of each plane
    planes = pascal_texture_planes_extended['car'].keys()
    visibilities = {}
    for pl_name in planes:
        plane = draw_plane_simple(pl_name, kpoints_2d, Color.WHITE, h, w, pascal_texture_planes_extended)

        plane_occl = draw_plane_occlusion(pl_name, planes, distances_3d, kpoints_2d, h, w,
                                          pascal_texture_planes_extended)

        absolute_area = get_plane_area(plane)
        occluded_area = get_plane_area(plane_occl)
        if occluded_area > 0.9 * absolute_area:
            visibilities[pl_name] = True
        else:
            visibilities[pl_name] = False

    return visibilities
