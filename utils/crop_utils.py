import numpy as np


def square_crop_from_bbox(image: np.ndarray, bbox: list, dataset: str = 'pascal'):
    """
    Return the image cropped according to given bounding box.
    The crop is a square of side=max(bbox_side_1, bbox_side_2).
    Image is padded with zeros if too small.
    """
    if dataset == 'pascal':
        image_h, image_w, _ = image.shape
        x_min, y_min, x_max, y_max = bbox
        side_x = x_max - x_min
        side_y = y_max - y_min
        major_side = max(side_x, side_y)
    else:
        image_h, image_w, _ = image.shape
        x_min, y_min, side_y, side_x, _ = bbox
        major_side = max(side_x, side_y)

    major_side *= 1.1

    center_x = x_min + side_x / 2  # defined as in https://github.com/geopavlakos/object3d/blob/master/demoCustom.m
    center_y = y_min + side_y / 2  # defined as in https://github.com/geopavlakos/object3d/blob/master/demoCustom.m
    scale = major_side / 200.      # defined as in https://github.com/geopavlakos/object3d/blob/master/demoCustom.m

    pad_x_before, pad_x_after, pad_y_before, pad_y_after = 0, 0, 0, 0

    new_x_min = int(center_x - major_side / 2.)
    if new_x_min < 0:
        pad_x_before = int(np.ceil(np.abs(new_x_min)))
        new_x_min = 0

    new_x_max = int(center_x + major_side / 2.) + pad_x_before
    if new_x_max > image_w:
        pad_x_after = int(np.ceil(np.abs(new_x_max - image_w)))
        new_x_max = image_w + pad_x_after

    new_y_min = int(center_y - major_side / 2.)
    if new_y_min < 0:
        pad_y_before = int(np.ceil(np.abs(new_y_min)))
        new_y_min = 0

    new_y_max = int(center_y + major_side / 2.) + pad_y_before
    if new_y_max > image_h:
        pad_y_after = int(np.ceil(np.abs(new_y_max - image_h)))
        new_y_max = image_h + pad_y_after

    padded = np.pad(image, [(pad_y_before, pad_y_after), (pad_x_before, pad_x_after), (0, 0)], mode='constant')
    crop = padded[new_y_min: new_y_max, new_x_min: new_x_max, :]
    return crop, (new_x_min, new_y_min), (pad_x_before, pad_y_before), \
           (pad_x_after, pad_y_after), (center_x, center_y), scale


def image_ref_to_crop_ref(kpoints_dict: dict, crop_tl: tuple, crop_pad: tuple, crop_shape: tuple, normalize: bool):
    """
    Shift keypoint coordinates from the image reference system to the crop one.

    :param kpoints_dict: Dictionary of keypoints to process
    :param crop_tl: Top-left (x, y) coordinates of bbox in image reference system
    :param crop_pad: Possible padding added to the image to crop properly
    :param crop_shape: Shape of the crop as (height, width)
    :param normalize: If True, coordinates are put in range [0, 1] dividing by crop size
    :return kpoints_dict_crop: Dictionary of keypoints to process in crop reference system
    """
    kpoints_crop = {}
    x_min, y_min = crop_tl
    x_pad, y_pad = crop_pad
    crop_h, crop_w, _ = crop_shape
    for (k_name, k_xy) in kpoints_dict.items():
        k_x, k_y = k_xy
        if np.all(np.asarray(k_xy) != -1):  # coords are valid
            # Shift joints coordinates from image reference system to crop's one
            k_x = k_xy[0] - x_min + x_pad
            k_y = k_xy[1] - y_min + y_pad

            #  It can happen that a keypoint is marked as visible even though it is
            #  occluded in reality. This means that when cropping the vehicle, the
            #  keypoint lies outside the bounding box. For the current moment it is
            #  discarded.
            if np.any(np.asarray([k_x, k_y]) < 0.) or (k_x >= crop_w) or (k_y >= crop_h):
                k_x, k_y = -1, -1
            elif normalize:
                # Put in range [0, 1] rescaling by crop size
                k_x = k_x / crop_w
                k_y = k_y / crop_h
        kpoints_crop[k_name] = [k_x, k_y]
    return kpoints_crop
