import cv2
import numpy as np

from utils.bounding_box import BoundingBox


def geodesic_distance(point1, point2):
    R = 6371.0  # approximate radius of earth
    dlon = np.radians(point1[1]) - np.radians(point2[1])
    dlat = np.radians(point1[0]) - np.radians(point2[0])
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(point2[0])) * \
        np.cos(np.radians(point1[0])) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance * 1000


def trajectories_to_meters(car_tracks, inv_homography_matrix, scale, shape, img_scale,
                           mode: str = 'traj'):
    GPS_coords = []
    if mode == 'inter':
        car_tracks = car_tracks[:, 2:]
    for curr_track in car_tracks:
        if mode == 'inter':
            mid_bottom = curr_track
        else:
            mid_bottom = np.asarray(BoundingBox(*curr_track[2:6] * img_scale,
                                                bounds=(0, shape[0] - 1, 0, shape[1] - 1),
                                                scale=scale).mid_bottom)
        mid_bottom = cv2.convertPointsToHomogeneous(mid_bottom[None, ...]).reshape(-1, 3)
        proj_mid_bottom = (inv_homography_matrix @ mid_bottom.transpose()).transpose()[0]
        proj_mid_bottom = proj_mid_bottom / proj_mid_bottom[2]
        GPS_coords.append(np.asarray(proj_mid_bottom[:-1]))
    GPS_coords = np.asarray(GPS_coords)

    # [top_left, bottom_right]
    tl_br_coords = [[np.min(GPS_coords[:, 0]), np.min(GPS_coords[:, 1])],
                    [np.max(GPS_coords[:, 0]), np.max(GPS_coords[:, 1])]]

    # [top_right, bottom_left]
    tr_bl_coord = [[tl_br_coords[1][0], tl_br_coords[0][1]],
                   [tl_br_coords[0][0], tl_br_coords[1][1]]]

    # Metric coordinates of bottom right point
    br_meter = []
    br_meter.append(geodesic_distance(tl_br_coords[1], tr_bl_coord[1]))
    br_meter.append(geodesic_distance(tl_br_coords[1], tr_bl_coord[0]))

    # GPS coordinates to meters
    meter_coords = np.zeros(GPS_coords.shape)
    lat_diff = tl_br_coords[1][0] - tl_br_coords[0][0]
    long_diff = tl_br_coords[1][1] - tl_br_coords[0][1]
    meter_coords[:, 0] = ((GPS_coords[:, 0] - tl_br_coords[0][0]) / lat_diff) * br_meter[0]
    meter_coords[:, 1] = ((GPS_coords[:, 1] - tl_br_coords[0][1]) / long_diff) * br_meter[1]

    return meter_coords