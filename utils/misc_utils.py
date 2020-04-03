from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from PIL.Image import Image


class Color:
    """
    Colors enumerator
    """
    BLACK = (0, 0, 0)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)


def load_yaml_file(file_path: Path):
    """
    Safe load of a `.yaml` file.`
    """
    if not isinstance(file_path, Path):
        raise ValueError('Please provide a valid Path.')

    if not file_path.is_file():
        raise FileNotFoundError(f'File {file_path} not found.')

    with file_path.open() as f:
        return yaml.safe_load(f)


def to_tensor(image: Union[Image, np.ndarray], max_range: int = 255):
    """
    Convert an image in range [0, `max_range`] to a torch tensor in range [-1, 1].
    :param image: np.ndarray or PIL image
    :param max_range: max value the image can assume
    :return: torch.FloatTensor
    """
    if isinstance(image, Image):
        image = np.asarray(image)
    image = np.float32(image)
    assert image.max() <= max_range
    image = image / max_range
    image = np.transpose(image, (2, 0, 1))
    image = image * 2. - 1.
    image = torch.from_numpy(image)
    return image