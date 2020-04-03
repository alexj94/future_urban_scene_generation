from pathlib import Path

import open3d as o3d

from utils.misc_utils import load_yaml_file


def load_ply_and_3d_kpoints(cad_root: Path, cad_idx: int, pascal_class='car'):
    """
    Load a .ply together with his 3D keypoints 
    (they must be placed in the same dir). 
    """

    # Update model and 3D keypoints
    ply_path = cad_root / f'pascal_{pascal_class}_cad_{cad_idx:03d}.ply'

    # Load 3D keypoints for current model
    yaml_file = ply_path.parent / (ply_path.stem + '.yaml')
    kpoints_3d = load_yaml_file(yaml_file)['kpoints_3d']

    mesh = o3d.io.read_triangle_mesh(str(ply_path))

    # Compute normal colors
    mesh.compute_vertex_normals()

    return mesh, kpoints_3d
