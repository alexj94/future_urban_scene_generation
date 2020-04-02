from open3d import draw_geometries_with_key_callbacks
from open3d import create_mesh_coordinate_frame
from open3d import read_triangle_mesh
import argparse
from pathlib import Path
import numpy as np


W = 1920 // 2
H = 1080 // 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_folder_path', type=Path, help='path to a folder with .ply files')
    args = parser.parse_args()
    # create origins
    geometries = []
    origin = create_mesh_coordinate_frame(size=0.25, origin=[0, 0, 0])
    geometries.append(origin)
    displace = 0.
    for i, ply_path in enumerate(args.ply_folder_path.iterdir()):
        print(ply_path)
        mesh = read_triangle_mesh(str(ply_path))
        mesh.compute_vertex_normals()
        mesh.transform(np.asarray([(1, 0, 0, displace),
                                   (0, 1, 0, displace),
                                   (0, 0, 1, 0),
                                   (0, 0, 0, 1)]))
        geometries.append(mesh)
        displace += 0.5

    draw_geometries_with_key_callbacks(geometries, {},
                                       width=W, height=H)
