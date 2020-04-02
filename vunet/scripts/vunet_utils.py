import cv2
import numpy as np
import torch

from utils.normalization import to_image


def get_filter_mask(x: torch.Tensor, threshold: int)-> torch.Tensor:
    """
    :param x: LAB Bx3xHxW
    :param threshold: [0, 255] value
    :return:
    """
    assert 0 <= threshold <= 255
    mask = []
    for el in x:
        el = to_image(el, from_LAB=True)
        el = cv2.cvtColor(el, cv2.COLOR_BGR2GRAY)  # we filter over just one channel
        el = el <= threshold
        el = np.stack([el] * 3, 0)
        mask.append(el)
    mask = np.stack(mask, 0)
    # mask = np.roll(mask, shift=-np.random.randint(0, 8), axis=2)
    mask = torch.from_numpy(mask.astype(np.uint8)).to(x.device)  # TODO check me
    return mask


def get_linear_val(step: int, start: int, end: int, start_val: float, end_val: float) -> float:
    if not end > start:
        raise ValueError('Provided interval is not valid.')

    linear_val = (step - start) / (end - start) * (end_val - start_val) + start_val

    clip_max = max((start_val, end_val))
    clip_min = min((start_val, end_val))
    return np.clip(linear_val, clip_min, clip_max)
