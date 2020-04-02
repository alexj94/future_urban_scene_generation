"""
    Pascal3DDataset training utility functions
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import open3d as o3d
from sklearn.metrics import confusion_matrix
from utils.misc_utils import load_yaml_file


def plot_confusion_matrix(root_dir, y_true, y_pred, normalize=False, title=None, cmap=plt.cm.get_cmap(name='Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Only use the labels that appear in the data
    classes = np.arange(0, 10)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if normalize:
        plt.savefig(str(root_dir / "conf_matrix_heatmap_norm.png"), dpi=500)
    else:
        plt.savefig(str(root_dir / "conf_matrix_heatmap.png"), dpi=500)
    plt.close(fig)


def make_exec_dir(args, now):
    for sub_d_name in ['train', 'train/checkpoints']:
        sub_d = args.res_dir / args.model / now / sub_d_name
        if not sub_d.is_dir():
            sub_d.mkdir(exist_ok=True, parents=True)
    dump_args(args, now)


def dump_args(args, now):
    args_file = open(str(args.res_dir / args.model / now / 'dump_args.yaml'), 'w')
    args_dict = {}
    for arg in vars(args):
        args_dict.update({str(arg): str(getattr(args, arg))})
    yaml.dump(args_dict, args_file, default_flow_style=False)


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
