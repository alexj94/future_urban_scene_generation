import cv2
import numpy as np
import torchvision.transforms as transforms
from skimage.feature import canny

from edgeconnect.config import Config
from utils.bounding_box import BoundingBox


def load_config(args):
    """
        Loading config file for EdgeConnect network
    """
    config = Config(str(args.checkpoints_dir / 'inpainting' / 'config.yml'))
    config.MODE = 2
    config.MODEL = args.inpaint_model if args.inpaint_model is not None else 3
    config.INPUT_SIZE = 0

    return config


def create_img_bbox(img, bbox, w, h):
    bbox_new_img = BoundingBox(*bbox, bounds=(0, w - 1, 0, h - 1), scale=1.3).xyxy

    curr_img_copy = img[bbox_new_img[1]:bbox_new_img[3], bbox_new_img[0]:bbox_new_img[2]]
    inpaint_h, inpaint_w, _ = curr_img_copy.shape
    bbox_x_min = bbox[0] - bbox_new_img[0]
    bbox_y_min = bbox[1] - bbox_new_img[1]
    bbox_inpaint = np.asarray([bbox_y_min, bbox_x_min, bbox_y_min + bbox[3], bbox_x_min + bbox[2]],
                              dtype=int)

    return curr_img_copy, bbox_inpaint, bbox_new_img


def create_inpaint_inputs_shape(config, img, mask, bbox_new_img, device):
    """
    Create inputs for EdgeConnect network (greyscale masked image + RGB masked image + edges)
    """

    resize_h = resize_w = 256

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    mask = cv2.dilate(mask, kernel, iterations=1)
    img = img[bbox_new_img[1]: bbox_new_img[3], bbox_new_img[0]: bbox_new_img[2]]
    img[np.where(mask == 255)[0], np.where(mask == 255)[1], :] = 255
    img = cv2.resize(img, resize_w, resize_h)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, resize_w, resize_h)
    mask = (mask > 0).astype(np.uint8) * 255
    mask_edge = (1 - mask / 255).astype(np.bool)
    edge = canny(img_gray, config.SIGMA, mask_edge)

    img = transforms.ToTensor()(img).to(device).unsqueeze(0).float()
    img_gray = transforms.ToTensor()(img_gray).to(device).unsqueeze(0).float()
    mask = transforms.ToTensor()(mask).to(device).unsqueeze(0).float()
    edge = transforms.ToTensor()(edge).to(device).unsqueeze(0).float()

    return img, img_gray, mask, edge
