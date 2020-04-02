# -*- coding: utf-8 -*-
"""
    KeypointDataset training utlity functions
"""
import warnings

import cv2
import numpy as np
import yaml
from matplotlib import cm

from utils.misc_utils import Color

_KP_NAMES = ['left_back_trunk', 'left_back_wheel', 'left_front_light',
             'left_front_wheel', 'right_back_trunk', 'right_back_wheel',
             'right_front_light', 'right_front_wheel', 'upper_left_rearwindow',
             'upper_left_windshield', 'upper_right_rearwindow',
             'upper_right_windshield']


def kpoints_dict_to_array(kpoints_dict: dict, dim: int = 2):
    """
    Convert keypoint dictionary to array
    """
    assert isinstance(kpoints_dict, dict)
    assert dim == 2 or dim == 3

    kpoints_list = []
    for key in _KP_NAMES:
        if key in kpoints_dict:
            kpoints_list.append(kpoints_dict[key])
        else:
            kpoints_list.append(np.full(shape=(dim,), fill_value=-1))

    kpoints_array = np.asarray(kpoints_list, dtype=float)
    return kpoints_array


def kpoints_array_to_dict(kpoints_array: np.ndarray):
    """
    Convert keypoint array to dictionary
    """
    assert len(kpoints_array.shape) == 2
    assert len(kpoints_array) == len(_KP_NAMES)
    assert 2 <= kpoints_array.shape[1] <= 3

    return {kp_name: val for kp_name, val in zip(_KP_NAMES, kpoints_array)}


def kpoint_to_heatmap(kpoint, shape, sigma):
    map_h, map_w = shape
    if np.all(kpoint > 0):
        x, y = kpoint
        x *= map_w
        y *= map_h
        xy_grid = np.mgrid[:map_w, :map_h].transpose(2, 1, 0)
        heatmap = np.exp(-np.sum((xy_grid - (x, y)) ** 2, axis=-1) / sigma ** 2)
        heatmap /= (heatmap.max() + np.finfo('float32').eps)
    else:
        heatmap = np.zeros((map_h, map_w))
    return heatmap


def heatmap_from_kpoints_array(kpoints_array, shape, sigma):
    heatmaps = []
    for kpoint in kpoints_array:
        heatmaps.append(kpoint_to_heatmap(kpoint, shape, sigma))
    return np.stack(heatmaps, axis=-1)


def get_maxima(heatmap_tensor, thresh):
    """
    Return the (x, y) coordinates of the argmax for each channel.
    Also, a second vector indicates if the value corresponding to the
    argmax is above or below the given threshold.

    :param heatmap_tensor: torch.Tensor of shape (batch_size, n_kpoints, height, width)
    :param thresh: Threshold used to filter false positives.
    """

    # todo: consider also threshold in keypoints maximum value

    if len(heatmap_tensor.shape) != 4:
        raise ValueError('Input tensor must have 4D shape (b, c, h, w).')

    # Find the max (x, y) for each heatmap
    heatmap_tensor = heatmap_tensor.to('cpu').numpy()
    batch_size, n_kpoints, h, w = heatmap_tensor.shape

    detected_kpoints = np.zeros(shape=(batch_size, n_kpoints, 2))
    # are_valid = np.zeros(shape=(batch_size, n_kpoints), dtype=np.bool)

    for b in range(batch_size):
        for c in range(n_kpoints):
            y, x = np.unravel_index(np.argmax(heatmap_tensor[b][c], axis=None), (h, w))
            detected_kpoints[b, c] = [x / w, y / h]
            # are_valid[b, c] = heatmap_tensor[b][c][y, x] > thresh

    return detected_kpoints


def random_blend_grid(true_blends, pred_blends):
    grid = []
    for i in range(0, len(true_blends)):
        grid.append(np.concatenate(true_blends[i], axis=2))
        grid.append(np.concatenate(pred_blends[i], axis=2))
    return grid


def to_colormap(heatmap_tensor, device, cmap = 'jet', cmap_range=(None, None)):
    if not isinstance(heatmap_tensor, np.ndarray):
        try:
            heatmap_tensor = heatmap_tensor.to('cpu').numpy()
        except RuntimeError:
            heatmap_tensor = heatmap_tensor.detach().to('cpu').numpy()

    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])

    heatmap_tensor = np.sum(heatmap_tensor, axis=1)

    output = []
    batch_size = heatmap_tensor.shape[0]
    for b in range(batch_size):
        rgb = cmap.to_rgba(heatmap_tensor[b])[:, :, :-1]
        output.append(rgb[:, :, ::-1])  # RGB to BGR

    output = np.asarray(output).astype(np.float32)
    output = output.transpose(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)

    return output


def make_exec_dir(args, now):
    for sub_d_name in ['train', 'train/checkpoints', 'train/train_blends', 'train/eval_blends']:
        sub_d = args.res_dir / now / sub_d_name
        if not sub_d.is_dir():
            sub_d.mkdir(exist_ok=True, parents=True)
    dump_args(args, now)


def dump_args(args, now):
    args_file = open(str(args.res_dir / now / 'dump_args.yaml'), 'w')
    args_dict = {}
    for arg in vars(args):
        args_dict.update({str(arg): str(getattr(args, arg))})
    yaml.dump(args_dict, args_file, default_flow_style=False)


def normalize_kpoints(kpoints_2d: np.ndarray, max_x: float, max_y: float):
    """
    Normalize keypoints range using provided max values
    """
    assert len(kpoints_2d.shape) == 2
    assert kpoints_2d.shape[1] == 2
    assert max_x > 0. and max_y > 0.

    kpoints_2d[:, 0] /= max_x
    kpoints_2d[:, 1] /= max_y

    if np.any(kpoints_2d > 1.):
        warnings.warn(f'Warning! Some keypoints have values greater than 1.0.'
                      f'Make sure that `max_x` and `max_y` are correct.')

    return kpoints_2d


def draw_kpoints(frame: np.ndarray, kpoints_2d: np.ndarray, radius: int=3,
                 color: Color=Color.BLUE, thickness: int=cv2.FILLED):
    """
    Draw 2D keypoints as circles on a given frame
    """
    assert len(kpoints_2d.shape) == 2
    assert kpoints_2d.shape[1] == 2

    for kpoint_2d in kpoints_2d:
        kpoint_2d = tuple([int(a) for a in kpoint_2d])
        cv2.circle(frame, kpoint_2d, radius, color, thickness)
