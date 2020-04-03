from typing import List
from typing import Union

import cv2
import numpy as np
import torch

from warp_learn.online_visibility import pascal_texture_planes


def get_planes(image: np.ndarray, src_kpoint_dict, pascal_class: str,
               planes_visibility):

    h, w = image.shape[:2]

    planes = []
    kpoints_planes = []
    visibilities = []
    for pl_name in pascal_texture_planes[pascal_class].keys():
        pl_kp_names = pascal_texture_planes[pascal_class][pl_name]

        src_p2d = np.asarray(
            [list(map(float, src_kpoint_dict[k])) for k in pl_kp_names])

        src_p2d[:, 0] *= w
        src_p2d[:, 1] *= h
        src_p2d = np.int32(src_p2d)

        src_mask = cv2.fillPoly(np.zeros_like(image), [src_p2d],
                                color=(1, 1, 1))
        src_to_warp = image * src_mask

        planes.append(src_to_warp)
        kpoints_planes.append(src_p2d)
        visibilities.append(planes_visibility[pl_name])

    return np.stack(planes, 0), kpoints_planes, np.stack(visibilities).astype(np.uint8)


def warp_unwarp_planes(src_planes: np.ndarray, src_planes_kpoints: List[np.ndarray],
                       dst_planes_kpoints: List[np.ndarray],
                       src_visibilities: np.ndarray, dst_visibilities: np.ndarray, pascal_class: str, pascal_texture_planes):
    planes_warped = np.zeros_like(src_planes, dtype=src_planes.dtype)
    planes_unwarped = np.zeros_like(src_planes, dtype=src_planes.dtype)

    keys = list(pascal_texture_planes[pascal_class].keys())
    symmetry_set = [keys.index('left'), keys.index('right')]

    for i, pl_name in enumerate(keys):
        """
        Conditions to skip:
        - pl not visible in src
        - pl not in symmetry and not visible in dst
        - pl in symmetry and neither one from the symmetry visible in dst

        """
        if not src_visibilities[i]:
            continue
        if i not in symmetry_set and not dst_visibilities[i]:
            continue
        if i in symmetry_set and 1 not in [dst_visibilities[j] for j in symmetry_set]:
            continue

        src_plane = src_planes[i]
        src_plane_kpoints = src_planes_kpoints[i]
        j = i
        if i in symmetry_set and not dst_visibilities[i]:
            j = symmetry_set[0] if i == symmetry_set[1] else symmetry_set[1]

        dst_plane_kpoints = dst_planes_kpoints[j]
        H12, _ = cv2.findHomography(src_plane_kpoints, dst_plane_kpoints)
        H21, _ = cv2.findHomography(dst_plane_kpoints, src_plane_kpoints)

        if H12 is not None and H21 is not None:
            h, w = src_planes[0].shape[0:2]
            src_warped = cv2.warpPerspective(src_plane, H12, dsize=(w, h))  # TODO I had to swap w and h here!
            src_unwarped = cv2.warpPerspective(src_warped, H21, dsize=(w, h))

            planes_warped[j] = src_warped
            planes_unwarped[i] = src_unwarped

    return planes_warped, planes_unwarped


def planes_to_torch(planes, to_LAB: bool):
    planes = [p for p in planes]
    if to_LAB:
        planes = [cv2.cvtColor(p, cv2.COLOR_BGR2LAB) for p in planes]
    planes = np.stack(planes)
    planes = np.float32(planes) / 255.
    planes = np.transpose(planes, (0, 3, 1, 2))
    planes = (torch.from_numpy(planes) - 0.5) / 0.5
    return planes


def to_image(x: Union[np.ndarray, torch.Tensor], from_LAB: bool):
    """
    Convert a tensor to a valid image for visualization.
    :param x: Input tensor is expected to be either LAB or BGR color space and to lie in range [-1, 1]
    :return x: Image BGR uint8 in range [0, 255]
    """
    assert len(x.shape) == 3, f'Unsupported image shape {x.shape}'

    try:
        x = x.to('cpu').detach().numpy()
        x = np.transpose(x, (1, 2, 0))
    except AttributeError:
        # Input tensor is already a numpy array
        pass

    x = (x + 1.) / 2 * 255
    x = np.clip(x, 0, 255)

    x = x.astype(np.uint8)

    if from_LAB:
        x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR)
    return x
