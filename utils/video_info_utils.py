from pathlib import Path

import numpy as np
import yaml


def parse_tracking_files(video_dir: Path, track_type: str, det_mode: str):
    """
    Parse CityFlow trajectory file
    """
    file_path = video_dir / 'mtsc' / f'mtsc_{track_type}_{det_mode}.txt'

    if not file_path.is_file():
        raise FileNotFoundError()

    content = np.loadtxt(file_path, delimiter=',')
    assert content.shape[1] == 10

    content = content[:, :-4]  # ignore last 4 columns

    return content


def parse_calibration_file(calibration_file: Path):
    """
    Read CityFlow calibration file
    """
    if not calibration_file.is_file():
        raise FileNotFoundError()

    with calibration_file.open('r') as content_file:

        content = yaml.safe_load(content_file)

        data_str = content['Homography matrix']

        matrix = []
        for i, row_str in enumerate(data_str.split(';')):
            cols = row_str.split(' ')
            row = [float(c) for c in cols]
            matrix.append(row)

        return np.asarray(matrix).astype(np.float64)
