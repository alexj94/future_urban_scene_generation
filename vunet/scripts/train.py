import argparse
import time
from pathlib import Path

import numpy as np
import progressbar
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasets.dataset_texture import TextureDatasetWithNormal
from utils.losses import compute_losses, compute_kl_loss
from utils.misc import init_worker
from utils.misc import suppress_random
from utils.normalization import to_image
from utils.saver import Saver
from vunet.model.perceptual_loss import PerceptualLayer
from vunet.model.vunet_fixed import Vunet_fix_res
from vunet.scripts.transfer import transfer_pass
from vunet.scripts.vunet_utils import get_linear_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('texture_dataset_dir', type=Path, help='Dataset directory')
    parser.add_argument('pascal_class', type=str, choices=['car', 'chair'], help='Pascal3D+ class')
    parser.add_argument('--vgg19_path', type=Path, help='Path to pre-trained VGG19 network')
    parser.add_argument('--model_path', type=Path, default=None, help='Pre-trained model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--ckpt_freq', type=int, default=5, help='Weights are saved every `ckpt_freq` epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
    parser.add_argument('--n_blocks', type=int, default=4, help='Number blocks in the network')
    parser.add_argument('--n_sampling_blocks', type=int, default=1, help='Number of sampling blocks')
    parser.add_argument('--up_mode', type=str, default='subpixel', choices=['nearest', 'subpixel', 'conv2d_t'])
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Resize factor to which images are resized')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Dropout probability used in training')
    parser.add_argument('--vgg_pool', type=str, default='max', choices=['max', 'avg'])
    parser.add_argument('--output_dir', type=Path, default=Path('/tmp'))
    parser.add_argument('--box_factor', type=int, default=2, help='Box factor for CompVis normalization')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--w_norm', action='store_true', help='Perform weight normalization')
    parser.add_argument('--vgg_upsample', action='store_true', help='Upsample to 224 before feeding VGG19')
    parser.add_argument('--demo', action='store_true', help='Use a subset of the dataset for debug')

    # Loss weights
    parser.add_argument('--w_content', type=float, default=1., help='Weight for perceptual loss')
    parser.add_argument('--w_gram', type=float, default=1., help='Weight for gram loss')
    parser.add_argument('--w_KL', type=float, default=1., help='Weight for KL loss')
    parser.add_argument('--augmentation', action='store_true',
                        help='Perform data augmentation')

    args = parser.parse_args()
    args.use_LAB = False

    # Helper object to save checkpoints, config, intermediate outputs
    saver = Saver(args)

    # Fix random seed to ease reproducibility
    suppress_random(seed=54321)
    dataset = TextureDatasetWithNormal(folder=args.texture_dataset_dir,
                                       resize_factor=args.resize_factor,
                                       demo_mode=args.demo,
                                       do_augmentation=args.augmentation, use_LAB=args.use_LAB)

    # Vunet
    vnet = Vunet_fix_res(args=args)
    vnet = vnet.to(args.device)

    perception_net = PerceptualLayer(vgg_path=args.vgg19_path, vgg_pool=args.vgg_pool, upsample=args.vgg_upsample)
    perception_net = perception_net.to(args.device)

    # Possibly load pre-trained weights
    if args.model_path is not None:
        vnet.load_state_dict(saver.load_state_dict(args.model_path))

    print(f'Net parameters: {np.sum([p.numel() for p in vnet.parameters()])}')
    print(dataset)
    time.sleep(0.5)

    # Instantiate data-loaders
    dl_config = dict(shuffle=True, drop_last=True, num_workers=4, worker_init_fn=init_worker)
    dl_train = DataLoader(dataset, batch_size=args.batch_size, **dl_config)
    dl_test = DataLoader(dataset, batch_size=32, **dl_config)

    optimizer = Adam(params=[p for p in vnet.parameters() if p.requires_grad], lr=args.lr, betas=(0.5, 0.9))

    n_batches = len(dl_train)
    widgets = [
        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {'total': n_batches}),
        ' ', progressbar.Bar(marker='-', left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('gram'), ' ',
        progressbar.DynamicMessage('gram_test'), ' ',
        progressbar.DynamicMessage('content'), ' ',
        progressbar.DynamicMessage('content_test'), ' ',
        progressbar.DynamicMessage('KL'), ' ',
        progressbar.DynamicMessage('KL_test'), ' ',
        progressbar.DynamicMessage('KL_w'), ' ',
        progressbar.DynamicMessage('LR'), ' ',
        progressbar.DynamicMessage('epoch')
    ]

    for epoch in range(args.epochs):
        progress = progressbar.ProgressBar(max_value=n_batches, widgets=widgets).start()

        # --------------------- Train one epoch on train set ------------------
        dataset.train()
        vnet.train()

        for i, data in enumerate(dl_train):

            # Get current learning rate according as recommended by original authors
            examples_seen = args.batch_size * (len(dl_train) * epoch + i)
            compvis_batch_size = 8  # batch size used in original implementation
            cur_lr = get_linear_val(step=examples_seen, start=1000 * compvis_batch_size,
                                    end=100000 * compvis_batch_size, start_val=args.lr, end_val=0.0)

            # w_KL = get_linear_val(step=examples_seen, start=10000 * compvis_batch_size,
            #                       end=200000 * compvis_batch_size, start_val=0.0, end_val=args.w_KL)
            w_KL = get_linear_val(step=examples_seen, start=1000 * compvis_batch_size,
                                  end=200000 * compvis_batch_size, start_val=0.0, end_val=args.w_KL)

            optimizer.param_groups[0]['lr'] = cur_lr

            x_1 = F.interpolate(data['src_image_masked'].to(args.device), 32)
            x_2 = F.interpolate(data['src_normal'].to(args.device), 32)

            x = torch.cat([x_1, x_2], 1)
            y_tilde = data['src_normal'].to(args.device)

            x_tilde, appearance_means, shape_means = vnet(y_tilde, x, mean_mode='mean_appearance')

            x_loss = data['src_image_masked'].to(args.device)

            losses = compute_losses(x_pred=x_tilde, x_true=x_loss, perception_net=perception_net)
            kl_loss = compute_kl_loss(appearance_means=appearance_means, shape_means=shape_means)
            loss = (losses['content'] * args.w_content +
                    losses['gram'] * args.w_gram +
                    kl_loss * w_KL)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.update(progress.value + 1, gram=losses['gram'] * args.w_gram, 
                            content=losses['content'] * args.w_content, KL=kl_loss * w_KL,
                            epoch=epoch + 1, LR=cur_lr, KL_w=w_KL)

        # ----------------- Evaluate metrics on validation set ----------------
        dataset.eval()
        vnet.eval()

        gram_loss_history = []
        content_loss_history = []
        KL_loss_history = []
        
        for data in dl_test:
            with torch.no_grad():
                x_1 = F.interpolate(data['src_image_masked'].to(args.device), 32)
                x_2 = F.interpolate(data['src_normal'].to(args.device), 32)

                x = torch.cat([x_1, x_2], 1)
                y_tilde = data['src_normal'].to(args.device)

                x_tilde, appearance_means, shape_means = vnet(y_tilde, x, mean_mode='mean_appearance')

                x_loss = data['src_image_masked'].to(args.device)

                losses = compute_losses(x_pred=x_tilde, x_true=x_loss, perception_net=perception_net)
                kl_loss = compute_kl_loss(appearance_means=appearance_means, shape_means=shape_means)
                loss = losses['gram'] * args.w_gram + kl_loss * w_KL + losses['content'] * args.w_content

                gram_loss_history.append(losses['gram'].item())
                content_loss_history.append(losses['content'].item())
                KL_loss_history.append(kl_loss.item())

                x_tilde_sampled = vnet(y_tilde, x=None, mean_mode='mean_shape')

        # ----------- Dump intermediate outputs on validation set -------------
        saver.dump_image(to_image(make_grid(x_tilde), from_LAB=args.use_LAB), epoch, 'reconstruction', split='eval')
        saver.dump_image(to_image(make_grid(x_tilde_sampled), from_LAB=args.use_LAB), epoch, 'sampling', split='eval')
        saver.dump_image(to_image(make_grid(data['src_image_masked']), from_LAB=args.use_LAB), epoch, 'target', split='eval')
        saver.dump_image(to_image(make_grid(data['src_log_image']), from_LAB=args.use_LAB), epoch, 'log', split='eval')
        saver.dump_image(to_image(make_grid(data['src_normal']), from_LAB=args.use_LAB), epoch, 'normal', split='eval')

        # ----------------------- Transfer pass -------------------------------
        with torch.no_grad():
            dataset.eval()
            vnet.eval()

            # For the last test batch, the appearance is conditioned on the
            #   first example of the batch and then transferred on all others
            x_1 = F.interpolate(data['src_image_masked'][0:1].to(args.device), 32)
            x_2 = F.interpolate(data['src_normal'][0:1].to(args.device), 32)

            x_input = torch.cat([x_1, x_2], 1)
            y_input = data['src_normal'][0:1].to(args.device)
            y_targets = data['src_normal'][1:].to(args.device)

            transfer_output = transfer_pass(vunet=vnet,
                                            appearance_enc_in=x_input,
                                            dest_shapes=y_targets)

            sticks_and_x_in = torch.cat([data['src_normal'][0:1], data['src_image_masked'][0:1]], 2)
            sticks_and_x_out = torch.cat([data['src_normal'][1:].to(args.device), transfer_output], dim=2).to('cpu')
            image = torch.cat([sticks_and_x_in, sticks_and_x_out], dim=0)
            saver.dump_image(to_image(make_grid(image), from_LAB=args.use_LAB), epoch, 'transfer', split='eval')

        # -------------------- Update progress bar ----------------------------
        progress.update(progress.max_value,
                        gram_test=np.average(gram_loss_history) * args.w_gram,
                        content_test=np.average(content_loss_history) * args.w_content,
                        KL_test=np.average(KL_loss_history) * w_KL, force=True)
        progress.finish()

        # -------------------- Save model weights -----------------------------
        if epoch % args.ckpt_freq == 0:
            saver.save(vnet, epoch)
