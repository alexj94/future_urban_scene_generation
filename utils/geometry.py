"""
    Miscellaneous geometric functions
"""
from typing import Sequence
from typing import Union

import cv2
import numpy as np
import open3d as o3d
import torch


def x_rot(alpha: float, clockwise: bool = False, pytorch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around X axis (default: counter-clockwise).

    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around X axis.
    """
    if pytorch:
        cx = torch.cos(alpha)
        sx = torch.sin(alpha)
    else:
        cx = np.cos(alpha)
        sx = np.sin(alpha)

    if clockwise:
        sx *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([one, zero, zero], dim=1),
                         torch.stack([zero, cx, -sx], dim=1),
                         torch.stack([zero, sx, cx], dim=1)], dim=0)
    else:
        rot = np.asarray([[1., 0., 0.],
                          [0., cx, -sx],
                          [0., sx, cx]], dtype=np.float32)
    return rot


def y_rot(alpha: float, clockwise: bool = False, pytorch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Y axis (default: counter-clockwise).

    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Y axis.
    """
    if pytorch:
        cy = torch.cos(alpha)
        sy = torch.sin(alpha)
    else:
        cy = np.cos(alpha)
        sy = np.sin(alpha)

    if clockwise:
        sy *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cy, zero, sy], dim=1),
                         torch.stack([zero, one, zero], dim=1),
                         torch.stack([-sy, zero, cy], dim=1)], dim=0)
    else:
        rot = np.asarray([[cy, 0., sy],
                          [0., 1., 0.],
                          [-sy, 0., cy]], dtype=np.float32)
    return rot


def z_rot(alpha: float, clockwise: bool = False, pytorch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Z axis (default: counter-clockwise).

    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Z axis.
    """
    if pytorch:
        cz = torch.cos(alpha)
        sz = torch.sin(alpha)
    else:
        cz = np.cos(alpha)
        sz = np.sin(alpha)

    if clockwise:
        sz *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cz, -sz, zero], dim=1),
                         torch.stack([sz, cz, zero], dim=1),
                         torch.stack([zero, zero, one], dim=1)], dim=0)
    else:
        rot = np.asarray([[cz, -sz, 0.],
                          [sz, cz, 0.],
                          [0., 0., 1.]], dtype=np.float32)

    return rot


def intrinsic_matrix(focal: float, cx: float, cy: float) -> np.ndarray:
    """
    Return intrinsics camera matrix with square pixel and no skew.

    :param focal: Focal length
    :param cx: X coordinate of principal point
    :param cy: Y coordinate of principal point
    :return K: intrinsics matrix of shape (3, 3)
    """
    return np.asarray([[focal, 0., cx],
                       [0., focal, cy],
                       [0., 0., 1.]])


def rototranslation_image(tvect, delta_t, rvect, delta_rot, K, dist, meshes, img):
    mesh_flat_draw = meshes @ delta_rot
    mesh_flat_draw = mesh_flat_draw + delta_t

    mesh2d, _ = cv2.projectPoints(mesh_flat_draw.astype(np.float64), rvect, tvect, K, dist)
    mesh2d = mesh2d.reshape(-1, 3, 2)
    cv2.drawContours(img, mesh2d.astype(int), -1, (0, 255, 0), 1)
    return img


def get_delta_t_vec(axis: str, t_value: float):
    assert axis in {'x', 'y', 'z'}
    t_vec = np.zeros(3)
    t_vec[['x', 'y', 'z'].index(axis)] = t_value
    return t_vec


def rotmat2azelrad(camera_coords):
    tx, ty, tz = camera_coords
    azimuth = np.degrees(np.arctan2(tx, tz))
    proj_coords = np.array([tx, 0, tz])
    camera_coords_norm = camera_coords / np.linalg.norm(camera_coords)
    proj_coords_norm = proj_coords / np.linalg.norm(proj_coords)
    elevation = np.degrees(np.arccos(np.dot(camera_coords_norm, proj_coords_norm)))

    while azimuth < 0:
        azimuth += 360
    while elevation < 0:
        elevation += 360
    radius = np.linalg.norm(camera_coords)

    return azimuth, elevation, radius


def create_sphere(radius: float, color: Sequence[int],
                  location: Union[list, np.ndarray]) -> o3d.geometry.TriangleMesh:
    """
    Create a spherical mesh with given location and color
    :param radius: Sphere radius
    :param color: Sphere color
    :param location: Sphere location in 3D space
    :return sphere: o3d.TriangleMesh of the sphere
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)

    sphere.paint_uniform_color(color)

    # Translate the sphere
    transformation = np.asarray([[1., 0., 0., location[0]],
                                 [0., 1., 0., location[1]],
                                 [0., 0., 1., location[2]],
                                 [0., 0., 0., 1.]])
    sphere.transform(transformation)

    return sphere


def create_plane_points():
    x = np.linspace(-2, 2, 10)
    x_points = []
    y_points = []
    for n, p in enumerate(x):
        x_points.append([x[0], p, 0])
        x_points.append([x[-1], p, 0])
        y_points.append([p, x[0], 0])
        y_points.append([p, x[-1], 0])
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)
    points = np.concatenate((x_points, y_points), axis=0)

    return points


def extrinsic_from_rodrigues(r_vect: np.ndarray,
                             t_vect: np.ndarray) -> np.ndarray:
    """
    Compute extrinsic from rotation and translation vectors encoded 
     in Rodrigues convention.
    """
    r_vect = r_vect.squeeze()
    t_vect = t_vect.squeeze()
    assert r_vect.shape == (3,)
    assert t_vect.shape == (3,)

    extrinsic = np.eye(4, dtype=r_vect.dtype)

    rot_matrix, _ = cv2.Rodrigues(r_vect)
    extrinsic[:3, :3] = rot_matrix
    extrinsic[:3, 3] = t_vect

    return extrinsic

