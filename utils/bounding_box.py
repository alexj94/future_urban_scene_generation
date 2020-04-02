# -*- coding: utf-8 -*-
"""
Bounding Box
"""
from typing import Union
from typing import Tuple

import numpy as np
import cv2


Scalar = Union[int, float]
Point2D = Union[Tuple[Scalar, Scalar], np.ndarray]
Color = Tuple[int, int, int]
Bounds = Tuple[Scalar, Scalar, Scalar, Scalar]


class BoundingBox:
    """
    Bounding Box
    """
    def __init__(self, x_min: Scalar, y_min: Scalar, w: Scalar, h: Scalar,
                 scale: float = None, bounds: Bounds = None):
        """
        Init BoundingBox instance

        :param x_min: X coordinate of top left coordinate
        :param y_min: Y coordinate of top left coordinate
        :param w: Width of the bounding box
        :param h: Height of the bounding box
        :param scale: Scale factor possibly used to resize the bounding box
        :param bounds: Image boundaries used to clip the tl and br corners
        """
        self.x_min = int(x_min)
        self.y_min = int(y_min)

        # self._check_valid(w, h)

        self.x_max = self.x_min + int(w)
        self.y_max = self.y_min + int(h)

        if scale is not None:
            self.rescale(scale)

        if bounds is not None:
            self.clip_to_bounds(bounds)

    def clip_to_bounds(self, bounds: Bounds):
        """
        Clip bounding box to stay inside bounds
        """
        x_min_b, x_max_b, y_min_b, y_max_b = bounds
        self.x_min = max(x_min_b, self.x_min)
        self.x_max = min(x_max_b, self.x_max)
        self.y_min = max(y_min_b, self.y_min)
        self.y_max = min(y_max_b, self.y_max)

    def contains(self, point: Point2D):
        """
        Check if point is contained in the bounding box
        """
        px, py = point
        contained = (self.x_min <= px <= self.x_max and
                     self.y_min <= py <= self.y_max)
        return contained

    def draw(self, frame: np.ndarray, color: Color, thickness: int = 2):
        """
        Draw bounding box on frame
        """
        top_left = (self.x_min, self.y_min)
        bottom_right = (self.x_max, self.y_max)

        cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    def rescale(self, scale: float):
        """
        Rescale bounding box according to `scale`
        """
        assert scale > 0.0

        new_width = self.width * scale
        delta_w = int(new_width - self.width)
        self.x_min -= delta_w // 2
        self.x_max += delta_w // 2

        new_height = self.height * scale
        delta_h = int(new_height - self.height)
        self.y_min -= delta_h // 2
        self.y_max += delta_h // 2

    @property
    def width(self):
        """
        Returns the width of the bounding box
        """
        return self.x_max - self.x_min

    @property
    def height(self):
        """
        Returns the height of the bounding box
        """
        return self.y_max - self.y_min

    @property
    def mid_bottom(self):
        """
        Calculate and return the middle bottom of the bbox
        """
        return self.x_min + self.width // 2, self.y_max

    @property
    def xyxy(self):
        """
        Return raw bounding box coords as (xmin, ymin, xmax, ymax)
        """
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xywh(self):
        """
        Return raw bounding box coords as (xmin, ymin, width, height)
        """
        return self.x_min, self.y_min, self.width, self.height

    @staticmethod
    def _check_valid(w: Scalar, h: Scalar):
        for value in [w, h]:
            assert value > 0
