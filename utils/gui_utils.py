# coding=utf-8
"""
Utils functions for GUI visualization
"""
import cv2
import numpy as np


def draw_help_on_frame(frame: np.ndarray, font_face=cv2.FONT_HERSHEY_PLAIN,
                       font_scale=1.5, font_color=(0, 0, 255), font_thick=2):
    """
    Draw help text on frame
    """
    cv2.rectangle(frame, (15, 60), (660, 220), (255, 255, 255), -1)
    cv2.putText(frame, "- Press 'SPACE' = next frame", (20, 90),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Single-click on bbox = visualize trajectory", (20, 120),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'DELETE' = clean frame from trajectories", (20, 150),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Double-click on bbox = projection inference", (20, 180),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'Q' = close window", (20, 210),
                font_face, font_scale, font_color, font_thick)


def draw_help_on_inter_frame(frame: np.ndarray, font_face=cv2.FONT_HERSHEY_PLAIN,
                             font_scale=1.5, font_color=(0, 0, 255), font_thick=2):
    """
    Draw help text on frame
    """
    cv2.putText(frame, "- Single-click on image = add future location", (10, 30),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'DELETE' = clean frame from trajectory", (10, 60),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'ENTER' to start inference", (10, 90),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'Q' = close window", (10, 120),
                font_face, font_scale, font_color, font_thick)


def draw_help_on_inf_frame(frame: np.ndarray, font_face=cv2.FONT_HERSHEY_PLAIN,
                       font_scale=1.5, font_color=(0, 0, 255), font_thick=2):
    """
    Draw help text on frame
    """
    cv2.rectangle(frame, (15, 60), (400, 220), (255, 255, 255), -1)
    cv2.putText(frame, "- Press 'W' = go forward", (20, 90),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'S' = go backward", (20, 120),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'A' = left rotation", (20, 150),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'D' = right rotation", (20, 180),
                font_face, font_scale, font_color, font_thick)
    cv2.putText(frame, "- Press 'Q' = close window", (20, 210),
                font_face, font_scale, font_color, font_thick)


def draw_trajectory(traj_meters: np.ndarray, cur_step: int = -1):
    assert len(traj_meters.shape) == 2
    assert cur_step < traj_meters.shape[0]

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)

    # We assume the mins are zero

    max_x, max_y = np.max(traj_meters, 0).astype(int)

    frame = np.zeros(shape=(max_y, max_x, 3), dtype=np.uint8)

    traj_meters = traj_meters.astype(np.int32)

    for t, pos in enumerate(traj_meters):

        color = red
        radius = 2
        thickness = 1
        if t == cur_step:
            color = green
            radius *= 2
            thickness = cv2.FILLED
        elif t > cur_step:
            color = blue

        x, y = pos
        cv2.circle(frame, (int(x), int(y)), radius=radius, color=color,
                   thickness=thickness)

    return frame


