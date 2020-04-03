from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19, VGG, make_layers, cfg


def subtract_mean(image_bgr: np.ndarray):
    imagenet_mean_BGR = [103.939, 116.779, 123.68]
    image_bgr = np.float32(image_bgr)
    image_bgr[..., 0] -= imagenet_mean_BGR[0]
    image_bgr[..., 1] -= imagenet_mean_BGR[1]
    image_bgr[..., 2] -= imagenet_mean_BGR[2]
    return image_bgr


class CaffeVGG19(VGG):
    """
    VGG19 with Caffe weights. Forward method differs from pytorch's VGG19, as features need to be permuted.
    Staticmethod caffe_to_torch can be used to translate weights between the two frameworks.
    Dropout is not discarded in pytorch's version, as it doesn't affect inference.
    """
    def __init__(self, weights_path: Path, pool_method: str = 'max'):
        super(CaffeVGG19, self).__init__(make_layers(cfg['E']))

        print(f'Loading pre-trained weights from {weights_path}... ', end='')
        self.load_state_dict(torch.load(weights_path))
        print('Done.')

        assert pool_method in ['max', 'avg']
        self.pool_method = pool_method
        if pool_method == 'avg':  # replace each max with and avg
            self.__max_to_avg()

    def forward(self, x):
        features = self.features(x)
        features = features.permute([0, 2, 3, 1]).contiguous()
        return self.classifier(features.view(len(features), -1))

    def __max_to_avg(self):
        features = []
        for f in self.features:
            if not isinstance(f, nn.MaxPool2d):
                features.append(f)
            else:
                features.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        self.features = nn.Sequential(*features)

    def __avg_to_max(self):
        features = []
        for f in self.features:
            if not isinstance(f, nn.AvgPool2d):
                features.append(f)
            else:
                features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.features = nn.Sequential(*features)

    @staticmethod
    def caffe_to_torch(load_path: Path, save_path: Path):
        """
        Translate weights from caffe to pytorch frameworks. load_path refers to a folder holding a .npy for each weight.
        """
        # load from path and sort them
        weight_caffe_paths = sorted(load_path.iterdir())
        weight_caffe = [np.load(w) for w in weight_caffe_paths]
        # load a pytorch standard vgg
        vgg = vgg19()
        # iterate over weights and translate them
        for w_1, w_2 in zip(weight_caffe, vgg.parameters()):
            print(f'torch->{w_1.shape}, caffe->{w_2.shape}')
            if len(w_1.shape) == 4:  # conv
                w_2.data = torch.from_numpy(np.transpose(w_1, [3, 2, 0, 1]))
            elif len(w_1.shape) == 2:  # linear
                w_2.data = torch.from_numpy(np.transpose(w_1, [1, 0]))
            else:  # bias
                w_2.data = torch.from_numpy(w_1)

        # save the vgg-19 architecture with new weights
        save_path.mkdir(exist_ok=True)
        torch.save(vgg.state_dict(), save_path / 'vgg19caffe.pth')

    def __call__(self, *args, **kwargs):
        return super(CaffeVGG19, self).__call__(*args, **kwargs)


if __name__ == '__main__':
    #CaffeVGG19.caffe_to_torch(Path('/home/luca/Desktop/vehicles/vunet/model/vgg19_weights_all'),
    #                          Path('/home/luca/Desktop/vehicles/vunet/model/'))
    vgg = CaffeVGG19(Path('/home/luca/Desktop/vehicles/vunet/model/vgg19caffe.pth'))
    vgg.eval()
    # test an image
    img_path = 'cauliflower.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    img = subtract_mean(img)
    img = np.expand_dims(img, 0)
    th_in = torch.from_numpy(img.transpose(0, 3, 1, 2))  # to channel first
    with torch.no_grad():
        class_new = vgg(th_in)

