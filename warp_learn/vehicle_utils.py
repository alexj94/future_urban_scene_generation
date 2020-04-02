import cv2
import numpy as np
import open3d as o3d

from utils.crop_utils import square_crop_from_bbox
from utils.keypoint_utils import kpoints_dict_to_array, kpoints_array_to_dict, normalize_kpoints
from warp_learn.online_visibility import compute_visibility
from warp_learn.planes_utils import get_planes
from warp_learn.render_open3d import get_rendered


def get_vehicle_information(ply: o3d.geometry.TriangleMesh, frame: np.ndarray,
                            extrinsic: np.ndarray, intrinsic: np.ndarray,
                            kpoints_2d_dict: dict, kpoints_3d_dict: dict):
    frame_h, frame_w = frame.shape[:2]

    sketch_normal, sketch_mask = get_rendered(ply, frame_w, frame_h,
                                              extrinsic, intrinsic)

    visibilities_dict = compute_visibility(extrinsic, intrinsic,
                                           kpoints_3d_dict, frame_h, frame_w)

    kpoints_2d = kpoints_dict_to_array(kpoints_2d_dict)
    # draw_kpoints(frame, kpoints_2d)
    kpoints_2d = normalize_kpoints(kpoints_2d, max_x=frame_w, max_y=frame_h)
    kpoints_2d_dict = kpoints_array_to_dict(kpoints_2d)
    planes, planes_kpoints, planes_visibilities = get_planes(frame,
                                                             kpoints_2d_dict,
                                                             'car',
                                                             visibilities_dict)

    return sketch_normal, sketch_mask, planes, planes_kpoints, planes_visibilities


def get_central_crop(bbox: list, frame: np.ndarray, icn_w: int, icn_h: int):
    """
    Get the central crop in a bbox. Central crop is a prior for the icn
    """

    img_bbox, xy_min, xy_pad, _, xy_center, scale = square_crop_from_bbox(frame, bbox)
    bbox_h, bbox_w, _ = img_bbox.shape
    img_bbox = cv2.resize(img_bbox, (icn_w, icn_h))

    # x_min, y_min, x_max, y_max = bbox
    # bbox = BoundingBox(x_min, y_min, abs(x_max - x_min), abs(y_max - y_min))
    # bbox.draw(frame, color=Color.GREEN)

    # GET CENTRAL CROP
    offset = int(icn_w * 0.1)
    src_central_crop = img_bbox[icn_h // 2 - offset:icn_h // 2 + offset, icn_w // 2 - offset:icn_w // 2 + offset].copy()

    src_central_crop = cv2.resize(src_central_crop, (icn_w, icn_h))
    return src_central_crop
