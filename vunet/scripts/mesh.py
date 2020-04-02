from math import radians, degrees
from typing import Tuple

import numpy as np
from open3d import Vector3dVector

from utils.geometry import y_rot
from utils.geometry import z_rot


def polar_to_cartesian(angle_y: float, angle_z: float, radius: float)->Tuple[float, float, float]:
    angle_y = radians(angle_y)
    angle_z = radians(angle_z)
    x = radius * np.sin(angle_y) * np.cos(angle_z)
    y = radius * np.sin(angle_y) * np.sin(angle_z)
    z = radius * np.cos(angle_y)
    return x, y, z


def cartesian_to_polar(x: float, y: float, z: float)->Tuple[float, float, float]:
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0.:
        angle_y = 0.
        angle_z = 0.
    else:
        angle_y = np.arccos(z / r)
        angle_z = np.arctan2(y, x)
    return degrees(angle_y), degrees(angle_z), r


class Mesh(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.zero_vertices = np.asarray(self.mesh.vertices).copy()
        # init properties
        self.__angle_y = 0.
        self.__angle_z = 0.
        self.__radius = 0.
        self.location = np.asarray([0, 0, 0], dtype=np.float32)
        # MMODE
        self.mmode = False
        self.__mmode_angle_xz = None # start xz angle
        self.__mmode_y = None
        self.__mmode_radius_xz = None

    def toggle_mmode(self):
        self.mmode = not self.mmode
        print(f'mmode set to {self.mmode}')
        if self.mmode:
            # save properties
            # we need the y coord, the radius on xz plane, the starting xz angle
            cartesian = polar_to_cartesian(self.angle_y, self.angle_z, self.radius)
            xz = [cartesian[0], cartesian[2]]

            self.mmode_angle_xz = degrees(np.arctan2(xz[1], xz[0]))
            self.__mmode_y = cartesian[1]
            self.__mmode_radius_xz = np.linalg.norm(xz)

        else:
            # set the polar to be here
            # get the three cords
            y = self.__mmode_y
            x = np.cos(radians(self.__mmode_angle_xz)) * self.__mmode_radius_xz
            z = np.sin(radians(self.__mmode_angle_xz)) * self.__mmode_radius_xz
            # get the corresponding polar
            angle_y, angle_z, radius = cartesian_to_polar(x, y, z)
            self.angle_y = angle_y
            self.angle_z = angle_z
            self.radius = radius

    @property
    def angle_y(self):
        return self.__angle_y

    @angle_y.setter
    def angle_y(self, angle_y):
        # ensure range
        self.__angle_y = float(np.clip(angle_y, -90, 90))

    @property
    def mmode_angle_xz(self):
        return self.__mmode_angle_xz

    @mmode_angle_xz.setter
    def mmode_angle_xz(self, mmode_angle_xz):
        # ensure range
        self.__mmode_angle_xz = float(np.clip(mmode_angle_xz, 0, 180))

    @property
    def angle_z(self):
        return self.__angle_z

    @angle_z.setter
    def angle_z(self, angle_z):
        if not self.mmode:
            # ensure range
            self.__angle_z = angle_z % 360

    @property
    def radius(self):
        return self.__radius

    @radius.setter
    def radius(self, radius):
        if not self.mmode:
            # ensure range
            self.__radius = max(0, radius)

    def update_mesh(self):

        if not self.mmode:
            angle_y = self.angle_y
            angle_z = self.angle_z
            radius = self.radius
        else:
            # get the three cords
            y = self.__mmode_y
            x = np.cos(radians(self.mmode_angle_xz)) * self.__mmode_radius_xz
            z = np.sin(radians(self.mmode_angle_xz)) * self.__mmode_radius_xz
            # get the corresponding polar
            angle_y, angle_z, radius = cartesian_to_polar(x, y, z)

        R = z_rot(radians(angle_z)) @ y_rot(radians(angle_y))
        #R[:, 0], R[:, 1] = R[:, 1].copy(), R[:, 0].copy()
        T = R[:, -1]
        T = T / np.linalg.norm(T) * radius
        #R[:, -1] *= - 1
        vertices = (R  @ self.zero_vertices.T).T
        vertices += T
        self.mesh.vertices = Vector3dVector(vertices)

    def get_extrinsic(self):
        R = z_rot(radians(self.angle_z)) @ y_rot(radians(self.angle_y))
        R[:, 0], R[:, 1] = R[:, 1].copy(), R[:, 0].copy()
        T = R[:, -1]
        T = T / np.linalg.norm(T) * self.radius
        R[:, -1] *= - 1
        ll= np.asarray([0, 0, 0, 1.0])
        RT = np.concatenate([R, T[:, None]], 1)
        RT = np.concatenate([RT, ll[None, :]], 0)
        return RT

