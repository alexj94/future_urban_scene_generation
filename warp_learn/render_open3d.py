import numpy as np
import open3d as o3d


def align_view(vis: o3d.visualization.VisualizerWithKeyCallback,
               intrinsic: np.ndarray, extrinsic: np.ndarray):

    assert extrinsic.shape == (3, 4) or extrinsic.shape == (4, 4)
    if extrinsic.shape == (3, 4):
        extrinsic = np.concatenate([extrinsic, np.asarray([[0, 0, 0, 1]])])
    assert intrinsic.shape == (3, 3)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]

    pinhole_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Get view controller intrinsics
    intrinsic = pinhole_params.intrinsic
    w, h = intrinsic.width, intrinsic.height
    cx, cy = intrinsic.get_principal_point()  # this must be left as they are
    intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)

    # Use current camera extrinsics to update view
    pinhole_params.extrinsic = extrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(pinhole_params)
    vis.poll_events()
    vis.update_renderer()


def get_rendered(model_ply, w, h, extrinsic, intrinsic):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
    vis.get_render_option().light_on = False
    vis.get_render_option().background_color = (0, 0, 0)
    vis.add_geometry(model_ply)

    model_ply.compute_vertex_normals()
    model_ply.vertex_colors = o3d.utility.Vector3dVector((np.asarray(model_ply.vertex_normals) + 1) / 2.)

    # transform geometry using ICP
    vis.poll_events()
    vis.update_renderer()
    align_view(vis, intrinsic, extrinsic)

    # Capture normal 2.5D sketch
    src_normal = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    src_normal = (src_normal * 255).astype(np.uint8)
    object_mask = np.all(src_normal == 0, axis=-1)
    vis.destroy_window()
    return src_normal, object_mask
