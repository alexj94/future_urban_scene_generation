from pathlib import Path
from time import time

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms.functional import normalize

from utils.bounding_box import BoundingBox
from utils.crop_utils import square_crop_from_bbox
from utils.geometry import extrinsic_from_rodrigues
from utils.geometry import get_delta_t_vec
from utils.geometry import z_rot
from utils.gps_utils import trajectories_to_meters
from utils.inpaint_utils import create_img_bbox
from utils.inpaint_utils import create_inpaint_inputs_shape
from utils.keypoint_utils import get_maxima
from utils.keypoint_utils import kpoints_array_to_dict
from utils.keypoint_utils import kpoints_dict_to_array
from utils.maskrcnn_utils import setup_cfg
from utils.misc_utils import to_tensor
from utils.pnp_utils import cpc_rodr_4_angles
from warp_learn.models import get_icn_inputs
from warp_learn.online_visibility import pascal_texture_planes
from warp_learn.planes_utils import to_image
from warp_learn.planes_utils import warp_unwarp_planes
from warp_learn.vehicle_utils import get_central_crop
from warp_learn.vehicle_utils import get_vehicle_information


# from detectron2.demo.predictor import VisualizationDemo
# from matplotlib import pyplot as plt
# from utils.gui_utils import draw_trajectory


def traj_test(args, cap, frame_id, frame, bboxes, trajectories, inv_homo_matrix, bbox_scale, img_scale,
              device, config, edge_model, inpaint_model, model_cad, model_kp, model_icn,
              model_VUnet, cads_ply, kpoints_dicts, inpaint_flag):
    if not inpaint_flag:
        back_frame = cv2.imread('/home/alessandro/Desktop/averaged_frame.png')
    h, w = frame.shape[:2]

    start_time = time()

    # Warp&Learn and VUnet list of future frames initialization
    result_frames_icn = np.zeros((6, h, w, 3), dtype=np.uint8)
    result_frames_vunet = np.zeros((6, h, w, 3), dtype=np.uint8)

    for i, bbox in enumerate(bboxes):
        with torch.no_grad():
            # bbox image for cad classification and keypoints localization
            img_bbox, xy_min, xy_pad, _, xy_center, scale = square_crop_from_bbox(frame, bbox)
            bbox_h, bbox_w, _ = img_bbox.shape
            img_bbox = cv2.resize(img_bbox, (256, 256))
            cadkp_images = transforms.ToTensor()(img_bbox)
            cadkp_images = normalize(cadkp_images.float(),
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            cadkp_images = cadkp_images.unsqueeze(0).to(device)

            # cad idx prediction on bbox
            out_cad = model_cad(cadkp_images)
            cad_idx = np.argmax(out_cad.to('cpu').numpy())
            print('\n#############################################################')
            print(f'Frame: {int(trajectories[i][0, 0])}, Car id: {int(trajectories[i][0, 1])}')
            print(f'Cad number: {cad_idx}')

            # keypoints localization on bbox
            out_kp = model_kp(cadkp_images)
            heatmap_pred = out_kp['heatmaps']
            heatmap_pred = F.interpolate(heatmap_pred[-1], (256, 256))
            keypoints_pred = get_maxima(heatmap_pred, 0.5)
            keypoints_pred = keypoints_pred.squeeze(axis=0)

            _____SCALE_F = 5

        if cad_idx is not None:
            ply = cads_ply[cad_idx]
            orig_kpoints_3d_dict = kpoints_dicts[cad_idx]

            kpoints3D_list = kpoints_dict_to_array(orig_kpoints_3d_dict, 3).astype(np.float32)

            kpoints3D_list *= _____SCALE_F
            orig_kpoints_3d_dict = kpoints_array_to_dict(kpoints3D_list)
            orig_vertices = np.asarray(ply.vertices).copy()

            # keypoint coordinates computation on real size of image
            for n in range(len(keypoints_pred)):
                keypoints_pred[n][0] = (keypoints_pred[n][0] * bbox_w) + xy_min[0] - xy_pad[0]
                keypoints_pred[n][1] = (keypoints_pred[n][1] * bbox_h) + xy_min[1] - xy_pad[1]

            # rototranslation parameters optimization (Levenberg-Marquardt)
            K = np.load(str(args.video_dir.parents[2] / 'intrinsic.npy'))
            dist = np.zeros((1, 5), dtype=np.float32)
            focals = np.asarray([K[0, 0], K[1, 1]], dtype=np.float)
            centers = np.asarray([K[0, 2], K[1, 2]], dtype=np.float)
            error_cpc, rvect_cpc, tvect_cpc = cpc_rodr_4_angles(focals, centers, keypoints_pred,
                                                                kpoints3D_list)
            rvect, tvect = rvect_cpc, tvect_cpc

            if inpaint_flag:
                # create inpainting inputs with mesh shape
                curr_img_copy = frame.copy()
                bbox_wh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
                _, _, bbox_new_img = create_img_bbox(curr_img_copy, bbox_wh, w, h)

                cfg = setup_cfg()
                demo = VisualizationDemo(cfg)
                img_input = curr_img_copy[bbox_new_img[1]:bbox_new_img[3], bbox_new_img[0]:bbox_new_img[2], :]
                predictions, _ = demo.run_on_image(img_input)
                masks = predictions['instances']._fields['pred_masks'].to('cpu').numpy()
                num_val_mask = []
                for cur_mask in masks:
                    num_val_mask.append(np.count_nonzero(cur_mask == True))
                mask_idx = np.argmax(np.asarray(num_val_mask))
                mask = masks[mask_idx]
                mask = np.where(mask == True, 255, 0).astype(np.uint8)

                inpaint_img, inpaint_img_gray, inpaint_mask, inpaint_edge = create_inpaint_inputs_shape(config, curr_img_copy, mask, bbox_new_img, device)

                # image inpainting on mesh shape
                edge = edge_model(inpaint_img_gray, inpaint_edge, inpaint_mask).detach()
                inpaint = inpaint_model(inpaint_img, edge, inpaint_mask)
                output_merged = (inpaint * inpaint_mask) + (inpaint_img * (1 - inpaint_mask))
                output = output_merged * 255.0
                output = output.permute(0, 2, 3, 1)
                output = output.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                img_output = cv2.resize(output, (bbox_new_img[2] - bbox_new_img[0],
                                                 bbox_new_img[3] - bbox_new_img[1]))

                # create Warp&Learn and VUnet inpainted vehicle crop image
                if i == 0:
                    inpainted_frame_icn = frame.copy()
                    inpainted_frame_vunet = frame.copy()
                else:
                    inpainted_frame_icn = result_frames_icn[0]
                    inpainted_frame_vunet = result_frames_vunet[0]
                inpainted_frame_icn[bbox_new_img[1]:bbox_new_img[3],
                                    bbox_new_img[0]:bbox_new_img[2]] = img_output
                inpainted_frame_vunet[bbox_new_img[1]:bbox_new_img[3],
                                      bbox_new_img[0]:bbox_new_img[2]] = img_output
                img_output_icn = inpainted_frame_icn
                img_output_vunet = inpainted_frame_vunet
            else:
                if i == 0:
                    img_output_icn = back_frame.copy()
                    img_output_vunet = back_frame.copy()
                else:
                    img_output_icn = result_frames_icn[0]
                    img_output_vunet = result_frames_vunet[0]

            try:
                ########################
                #      Warp&Learn      #
                ########################

                icn_w, icn_h = 256, 256
                extrinsic = extrinsic_from_rodrigues(rvect, tvect)

                central_crop = get_central_crop(bbox, frame, icn_w, icn_h)

                # get planes and stuff
                src_sketch_normal, src_sketch_mask, src_planes, src_planes_kpoints, src_planes_visibilities = get_vehicle_information(
                    ply, frame, extrinsic, K, kpoints_array_to_dict(keypoints_pred), orig_kpoints_3d_dict)
                dst_sketch_normal, dst_sketch_mask, dst_planes, dst_planes_kpoints, dst_planes_visibilities = get_vehicle_information(
                    ply, frame, extrinsic, K, kpoints_array_to_dict(keypoints_pred), orig_kpoints_3d_dict)

                # warp planes
                planes_warped, _ = warp_unwarp_planes(src_planes, src_planes_kpoints,
                                                      dst_planes_kpoints, src_planes_visibilities,
                                                      dst_planes_visibilities, 'car',
                                                      pascal_texture_planes)
                dst_sketch_mask = np.logical_not(dst_sketch_mask)

                # get cropped inputs
                icn_input, crop_info = get_icn_inputs(planes_warped, dst_sketch_normal, dst_sketch_mask,
                                                      central_crop, icn_w, icn_h)

                # forward
                net_image = to_image(model_icn(icn_input.to(device))[0], from_LAB=True)

                # revert and stitch
                crop_size_orig = crop_info['crop_size_orig']
                pad_xy_before = crop_info['pad_xy_before']
                pad_xy_after = crop_info['pad_xy_after']
                crop_xy_min = crop_info['crop_xy_min']

                crop_inv = cv2.resize(net_image, crop_size_orig[::-1])
                crop_inv = crop_inv[pad_xy_before[1]:crop_inv.shape[0] - pad_xy_after[1],
                                    pad_xy_before[0]:crop_inv.shape[1] - pad_xy_after[0]]

                out_frame = np.zeros_like(frame)
                out_frame[crop_xy_min[1]: crop_xy_min[1] + crop_inv.shape[0],
                          crop_xy_min[0]: crop_xy_min[0] + crop_inv.shape[1]] = crop_inv
                img_output_icn[dst_sketch_mask] = out_frame[dst_sketch_mask]
                result_frames_icn[0] = img_output_icn

                ####################
                #       VUnet      #
                ####################

                # create VUnet inputs
                src_sketch_mask_bbox = np.bitwise_not(src_sketch_mask)[..., np.newaxis] * frame
                ys, xs = np.nonzero(np.logical_not(src_sketch_mask))
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)
                src_sketch_mask_bbox, _, _, _, _, _ = square_crop_from_bbox(src_sketch_mask_bbox,
                                                                            [x_min, y_min, x_max,
                                                                             y_max])
                src_sketch_normal_bbox, _, _, _, _, _ = square_crop_from_bbox(src_sketch_normal,
                                                                              [x_min, y_min, x_max,
                                                                               y_max])
                dst_sketch_normal_bbox, _, _, _, _, _ = square_crop_from_bbox(dst_sketch_normal,
                                                                              [x_min, y_min, x_max,
                                                                               y_max])
                src_sketch_mask_bbox = cv2.resize(src_sketch_mask_bbox, (icn_w, icn_h))
                src_sketch_normal_bbox = cv2.resize(src_sketch_normal_bbox, (icn_w, icn_h))
                dst_sketch_normal_bbox = cv2.resize(dst_sketch_normal_bbox, (icn_w, icn_h))

                mask = np.all(src_sketch_normal_bbox == 0, axis=-1)
                src_sketch_mask_bbox[mask] = 255
                x_1 = F.interpolate(to_tensor(src_sketch_mask_bbox).unsqueeze(0), 256)
                x_2 = F.interpolate(to_tensor(src_sketch_normal_bbox[..., ::-1]).unsqueeze(0), 256)
                x = torch.cat([x_1, x_2], 1).to(args.device)
                y_tilde = to_tensor(dst_sketch_normal_bbox[..., ::-1]).unsqueeze(0).to(args.device)

                # forward
                output_enc_up, skips_enc_up = model_VUnet.forward_enc_up(x)
                mu_app, z_app = model_VUnet.forward_enc_down(output_enc_up, skips_enc_up)
                output_dec_up, skips_dec_up = model_VUnet.forward_dec_up(y_tilde)
                net_image, _, _ = model_VUnet.forward_dec_down(output_dec_up, skips_dec_up, mu_app)
                net_image = to_image(net_image[0], from_LAB=False)

                # revert and stitch
                crop_size_orig = crop_info['crop_size_orig']
                pad_xy_before = crop_info['pad_xy_before']
                pad_xy_after = crop_info['pad_xy_after']
                crop_xy_min = crop_info['crop_xy_min']

                crop_inv = cv2.resize(net_image, crop_size_orig[::-1])
                crop_inv = crop_inv[pad_xy_before[1]:crop_inv.shape[0] - pad_xy_after[1],
                           pad_xy_before[0]:crop_inv.shape[1] - pad_xy_after[0]]

                out_frame = np.zeros_like(frame)
                out_frame[crop_xy_min[1]: crop_xy_min[1] + crop_inv.shape[0],
                          crop_xy_min[0]: crop_xy_min[0] + crop_inv.shape[1]] = crop_inv
                img_output_vunet[dst_sketch_mask] = out_frame[dst_sketch_mask]
                result_frames_vunet[0] = img_output_vunet

            except:
                continue

            # trajectory to meter transformation
            meter_coords = trajectories_to_meters(trajectories[i], inv_homo_matrix, bbox_scale,
                                                  [w, h], img_scale)

            x_start, y_start = meter_coords[0]
            delta_x = np.mean(meter_coords[1:20, 0] - x_start)
            delta_y = np.mean(meter_coords[1:20, 1] - y_start)
            theta_start = np.arctan2(delta_y, delta_x)
            print(f'Theta start: {np.degrees(theta_start)}')

            # inference for every future trajectory vehicle position within 1 second
            curr_frame_id = frame_id
            for n, cur_pos in enumerate(meter_coords[1:], 1):
                # rotation and translation computation from trajectory
                distance = np.linalg.norm(meter_coords[0] - cur_pos)
                print(f'Distance: {distance}')

                x_cur, y_cur = cur_pos
                delta_x = x_cur - x_start
                delta_y = y_cur - y_start

                theta = np.arctan2(delta_y, delta_x) - theta_start
                print(f'Theta: {np.degrees(theta)}')

                delta_t = get_delta_t_vec('y', -distance)

                if 1 < n < len(meter_coords[1:]) - 1:
                    cur_delta_x = x_cur - meter_coords[n - 1, 0]
                    cur_delta_y = y_cur - meter_coords[n - 1, 1]
                    next_delta_x = meter_coords[n + 1, 0] - x_cur
                    next_delta_y = meter_coords[n + 1, 1] - y_cur
                    cur_theta = np.degrees(np.arctan2(cur_delta_y, cur_delta_x))
                    next_theta = np.degrees(np.arctan2(next_delta_y, next_delta_x))
                    theta_diff = cur_theta - next_theta
                    print(f'Instant theta: {theta_diff}')
                    if -20 < theta_diff < 20:
                        tr = delta_t @ z_rot(theta)
                    else:
                        tr = delta_t @ z_rot(0)
                else:
                    if -20 < np.degrees(theta) < 20:
                        tr = delta_t @ z_rot(theta)
                    else:
                        tr = delta_t @ z_rot(0)

                try:
                    if inpaint_flag:
                        # get next frame
                        curr_frame_id += 2
                        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_id - 1)
                        ret, cur_frame = cap.read()
                        if not ret:
                            break
                        cur_frame = cv2.resize(cur_frame, (1280, 720))

                        # create inpainting inputs with mesh shape
                        curr_img_copy = cur_frame.copy()
                        bbox_wh = BoundingBox(*trajectories[i][n, 2:6] * img_scale,
                                              bounds=(0, w - 1, 0, h - 1), scale=bbox_scale).xywh
                        _, _, bbox_new_img = create_img_bbox(curr_img_copy, bbox_wh, w, h)

                        img_input = curr_img_copy[bbox_new_img[1]:bbox_new_img[3],
                                    bbox_new_img[0]:bbox_new_img[2], :]
                        predictions, _ = demo.run_on_image(img_input)
                        masks = predictions['instances']._fields['pred_masks'].to('cpu').numpy()
                        num_val_mask = []
                        for cur_mask in masks:
                            num_val_mask.append(np.count_nonzero(cur_mask == True))
                        mask_idx = np.argmax(np.asarray(num_val_mask))
                        mask = masks[mask_idx]
                        mask = np.where(mask == True, 255, 0).astype(np.uint8)

                        inpaint_img, inpaint_img_gray, inpaint_mask, inpaint_edge = create_inpaint_inputs_shape(config, curr_img_copy, mask, bbox_new_img, device)

                        # image inpainting on mesh shape
                        edge = edge_model(inpaint_img_gray, inpaint_edge, inpaint_mask).detach()
                        inpaint = inpaint_model(inpaint_img, edge, inpaint_mask)
                        output_merged = (inpaint * inpaint_mask) + (inpaint_img * (1 - inpaint_mask))
                        output = output_merged * 255.0
                        output = output.permute(0, 2, 3, 1)
                        output = output.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                        img_output = cv2.resize(output, (bbox_new_img[2] - bbox_new_img[0],
                                                         bbox_new_img[3] - bbox_new_img[1]))

                        # create Warp&Learn and VUnet inpainted vehicle crop image
                        if i == 0:
                            inpainted_frame_icn = cur_frame.copy()
                            inpainted_frame_vunet = cur_frame.copy()
                        else:
                            inpainted_frame_icn = result_frames_icn[n]
                            inpainted_frame_vunet = result_frames_vunet[n]
                        inpainted_frame_icn[bbox_new_img[1]:bbox_new_img[3],
                                            bbox_new_img[0]:bbox_new_img[2]] = img_output
                        inpainted_frame_vunet[bbox_new_img[1]:bbox_new_img[3],
                                              bbox_new_img[0]:bbox_new_img[2]] = img_output
                        img_output_icn = inpainted_frame_icn
                        img_output_vunet = inpainted_frame_vunet
                    else:
                        if i == 0:
                            img_output_icn = back_frame.copy()
                            img_output_vunet = back_frame.copy()
                        else:
                            img_output_icn = result_frames_icn[n]
                            img_output_vunet = result_frames_vunet[n]

                    # rotate and translate 2D and 3D keypoints
                    kpoints_3d_dict = orig_kpoints_3d_dict.copy()
                    for k, v in kpoints_3d_dict.items():
                        kpoints_3d_dict[k] = v @ z_rot(theta) + tr
                    ply.vertices = o3d.Vector3dVector(orig_vertices @ z_rot(theta) + tr)
                    kpoints_2d_next, _ = cv2.projectPoints(
                        kpoints_dict_to_array(kpoints_3d_dict, dim=3),
                        rvect, tvect, K, dist)
                    kpoints_2d_next = kpoints_2d_next.squeeze(1)

                    ########################
                    #      Warp&Learn      #
                    ########################

                    # get planes and stuff
                    dst_sketch_normal, dst_sketch_mask, dst_planes, dst_planes_kpoints, dst_planes_visibilities = get_vehicle_information(
                        ply, frame, extrinsic, K, kpoints_array_to_dict(kpoints_2d_next),
                        kpoints_3d_dict)

                    # warp planes
                    planes_warped, _ = warp_unwarp_planes(src_planes, src_planes_kpoints,
                                                          dst_planes_kpoints,
                                                          src_planes_visibilities,
                                                          dst_planes_visibilities, 'car',
                                                          pascal_texture_planes)
                    dst_sketch_mask = np.logical_not(dst_sketch_mask)

                    # get cropped inputs
                    icn_input, crop_info = get_icn_inputs(planes_warped, dst_sketch_normal,
                                                          dst_sketch_mask,
                                                          central_crop, icn_w, icn_h)
                    # forward
                    net_image = to_image(model_icn(icn_input.to(device))[0], from_LAB=True)

                    # revert and stitch
                    crop_size_orig = crop_info['crop_size_orig']
                    pad_xy_before = crop_info['pad_xy_before']
                    pad_xy_after = crop_info['pad_xy_after']
                    crop_xy_min = crop_info['crop_xy_min']

                    crop_inv = cv2.resize(net_image, crop_size_orig[::-1])
                    crop_inv = crop_inv[pad_xy_before[1]:crop_inv.shape[0] - pad_xy_after[1],
                                        pad_xy_before[0]:crop_inv.shape[1] - pad_xy_after[0]]

                    out_frame = np.zeros_like(frame)
                    out_frame[crop_xy_min[1]: crop_xy_min[1] + crop_inv.shape[0],
                              crop_xy_min[0]: crop_xy_min[0] + crop_inv.shape[1]] = crop_inv
                    img_output_icn[dst_sketch_mask] = out_frame[dst_sketch_mask]
                    result_frames_icn[n] = img_output_icn

                    ####################
                    #       VUnet      #
                    ####################

                    # create VUnet inputs
                    ys, xs = np.nonzero(dst_sketch_mask)
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    dst_sketch_normal_bbox, _, _, _, _, _ = square_crop_from_bbox(dst_sketch_normal,
                                                                                  [x_min, y_min,
                                                                                   x_max, y_max])
                    dst_sketch_normal_bbox = cv2.resize(dst_sketch_normal_bbox, (icn_w, icn_h))
                    y_tilde = to_tensor(dst_sketch_normal_bbox[..., ::-1]).unsqueeze(0).to(args.device)

                    # forward
                    output_dec_up, skips_dec_up = model_VUnet.forward_dec_up(y_tilde)
                    net_image, _, _ = model_VUnet.forward_dec_down(output_dec_up, skips_dec_up, mu_app)
                    net_image = to_image(net_image[0], from_LAB=False)

                    # revert and stitch
                    crop_size_orig = crop_info['crop_size_orig']
                    pad_xy_before = crop_info['pad_xy_before']
                    pad_xy_after = crop_info['pad_xy_after']
                    crop_xy_min = crop_info['crop_xy_min']

                    crop_inv = cv2.resize(net_image, crop_size_orig[::-1])
                    crop_inv = crop_inv[pad_xy_before[1]:crop_inv.shape[0] - pad_xy_after[1],
                                        pad_xy_before[0]:crop_inv.shape[1] - pad_xy_after[0]]

                    out_frame = np.zeros_like(frame)
                    out_frame[crop_xy_min[1]: crop_xy_min[1] + crop_inv.shape[0],
                    crop_xy_min[0]: crop_xy_min[0] + crop_inv.shape[1]] = crop_inv
                    img_output_vunet[dst_sketch_mask] = out_frame[dst_sketch_mask]
                    result_frames_vunet[n] = img_output_vunet

                except:
                    break

                # frame_traj = draw_trajectory(meter_coords, cur_step=n)
                # cv2.imshow('traj', frame_traj)
                # cv2.waitKey(0)

            # cv2.destroyAllWindows()

            # reset video capture and ply vertices
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ply.vertices = o3d.Vector3dVector(orig_vertices)

    end_time = time()
    print(f"Prediction at 0.6s in the future of {len(bboxes)} vehicles "
          f"took {end_time - start_time} seconds!")

    warp_learn_res_dir = Path(f'./results/warp&learn/'
                              f'{args.video_dir._cparts[-2]}_{args.video_dir._cparts[-1]}')
    if not warp_learn_res_dir.is_dir():
        warp_learn_res_dir.mkdir(parents=True, exist_ok=True)
    vunet_res_dir = Path(f'./results/vunet/'
                              f'{args.video_dir._cparts[-2]}_{args.video_dir._cparts[-1]}')
    if not vunet_res_dir.is_dir():
        vunet_res_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = [frame_id, frame_id + 2, frame_id + 4, frame_id + 6, frame_id + 8, frame_id + 10]
    for i, id in enumerate(frame_ids):
        cv2.imwrite(str(warp_learn_res_dir / f'{id:04}.png'), result_frames_icn[i])
        cv2.imwrite(str(vunet_res_dir / f'{id:04}.png'), result_frames_vunet[i])

    return cap
