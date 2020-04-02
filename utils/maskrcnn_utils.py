from detectron2.config import get_cfg


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file('/home/alessandro/Desktop/future_scene_synthesis/detectron2/'
                        'configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    cfg.merge_from_list(['MODEL.WEIGHTS',
                         '/home/alessandro/Desktop/checkpoints/model_final_a3ec72.pkl'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg