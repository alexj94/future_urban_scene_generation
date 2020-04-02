# -*- coding: utf-8 -*-
"""
Miscellaneous utils
"""
from pathlib import Path

import yaml


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
