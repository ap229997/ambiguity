# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import itertools
import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms

from zoedepth.utils.config import change_dataset

from .ddad import get_ddad_loader
from .diml_indoor_test import get_diml_indoor_loader
from .diml_outdoor_test import get_diml_outdoor_loader
from .diode import get_diode_loader
from .hypersim import get_hypersim_loader
from .ibims import get_ibims_loader
from .sun_rgbd_loader import get_sunrgbd_loader
from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader

from .preprocess import CropParams, get_white_border, get_black_border


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)

        if mode == 'train':

            self.training_samples = DataLoadPreprocess(
                config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=False, #True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler,
                                   worker_init_fn=self.training_samples.reset_rng)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation
                # only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class EvalDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)

        self.testing_samples = EvalDataset(
            config, mode, transform=transform)
        self.data = DataLoader(self.testing_samples,
                                batch_size=1, shuffle=False, num_workers=self.config.workers) # current code only works with bz=1 for eval


def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


class MixedNYUKITTI(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        nyu_conf = change_dataset(edict(config), 'nyu')
        kitti_conf = change_dataset(edict(config), 'kitti')

        # make nyu default for testing
        self.config = config = nyu_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            nyu_loader = DepthDataLoader(
                nyu_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            kitti_loader = DepthDataLoader(
                kitti_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
        else:
            self.data = DepthDataLoader(nyu_conf, mode, device=device).data


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        
        self.local_split = False
        if config.dataset == 'nyu': # train on local splits
            train_file_name = config.filenames_local_train
            online_eval_file_name = config.filenames_local_val
            self.local_split = True
        else:
            train_file_name = config.filenames_file
            online_eval_file_name = config.filenames_file_eval
        
        if mode == 'online_eval':
            with open(online_eval_file_name, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(train_file_name, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()
        self.reset_rng()

    def reset_rng(self, seed=0):
        # print(f'Reseting RNG with seed={seed}') 
        self.rng = np.random.RandomState(seed)

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        if self.mode == 'online_eval':
            idx = idx % len(self.filenames) # this is useful when running evaluation on multiple crops
        
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}
        
        flipped = False
        if self.mode == 'train':
            if self.config.dataset == 'kitti' and self.config.use_right and self.rng.random() > 0.5:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[1]))

            image = self.reader.open(image_path)
            depth_gt = self.reader.open(depth_path)
            w, h = image.size

            if self.config.do_kb_crop:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.dataset == 'nyu' and self.config.avoid_boundary:
                # print("Avoiding Blank Boundaries!")
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)


            if self.config.aug and self.config.do_random_rotate:
                random_angle = (self.rng.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.config.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0
            crop_box = np.array([0, 640, 0, 480], dtype=np.float32)
            if self.config.aug and self.config.random_crop:
                image, depth_gt, crop_box = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)
            # crop_box format: [x1, x2, y1, y2]
            if self.config.aug and self.config.random_translate:
                # print("Random Translation!")
                image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)

            # there is flipping in here as well, change bboxes accordingly
            image, depth_gt, crop_box, flipped, scaled = self.train_preprocess(image, depth_gt, crop_box, size=(w,h), 
                                        use_flip_aug=self.config.get('use_flip_aug', False), use_scale_aug=self.config.get('use_scale_aug', False))
            
            if self.config.get('use_scale_aug', False) and not scaled:
                # take center crop of size (512, 384)
                top_margin = int((image.shape[0] - 384) / 2)
                left_margin = int((image.shape[1] - 512) / 2)
                image = image[top_margin:top_margin + 384, left_margin:left_margin + 512, :]
                depth_gt = depth_gt[top_margin:top_margin + 384, left_margin:left_margin + 512].reshape((384,512,1))
                crop_box = (crop_box * 0.8) # fixed size (384, 512) input to the network
            
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                      'mask': mask, 'crop_box': crop_box, **sample}

        else:
            if self.mode == 'online_eval' and (not self.local_split):
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0]))
            image = np.asarray(self.reader.open(image_path),
                               dtype=np.float32) / 255.0
            h, w = image.shape[:2]

            if self.mode == 'online_eval':
                if self.local_split:
                    gt_path = self.config.gt_path
                else:
                    gt_path = self.config.gt_path_eval
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1]))
                has_valid_depth = False
                try:
                    depth_gt = self.reader.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))
                # print (image_path)

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.config.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

                    mask = np.logical_and(
                        depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                else:
                    mask = False

            if self.config.do_kb_crop:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]
            
            crop_box = np.array([0, 640, 0, 480], dtype=np.float32)
            if self.config.test_on_crop and self.config.aug and self.config.random_crop:
                image, depth_gt, crop_box = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)

            if self.config.get('use_scale_aug', False):
                # take center crop of size (512, 384)
                top_margin = int((image.shape[0] - 384) / 2)
                left_margin = int((image.shape[1] - 512) / 2)
                image = image[top_margin:top_margin + 384, left_margin:left_margin + 512, :]
                depth_gt = depth_gt[top_margin:top_margin + 384, left_margin:left_margin + 512].reshape((384,512,1))
                crop_box = (crop_box * 0.8) # fixed size (384, 512) input to the network

            if self.mode == 'online_eval':
                sample = {'image': image, 
                          'depth': depth_gt, 
                          'focal': focal,
                          'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 
                          'depth_path': sample_path.split()[1],
                          'mask': mask,
                          'crop_box': crop_box
                         }
            else:
                sample = {'image': image, 'focal': focal}

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)
        
        # change crop_box to [x1, y1, x2, y2] format
        if 'crop_box' in sample:
            crop_box = sample['crop_box']
            sample['crop_box'] = np.array([crop_box[0], crop_box[2], crop_box[1], crop_box[3]], dtype=np.float32)
            bbox = sample['crop_box'].astype(np.int16)

            # taken from toolbox/camera_params.m of NYU dataset
            intrx = np.eye(3)
            intrx[0,0] = focal
            intrx[1,1] = focal
            if self.config.get('use_gt_principal', False):
                intrx[0,2] = 325.58
                intrx[1,2] = 253.73
                if flipped:
                    intrx[0,2] = 640 - intrx[0,2] # doesn't make much of a difference, this is for NYU only

            pos_enc = self.config.get('pos_enc', None)
            if pos_enc is not None and pos_enc == 'center+corner_latent': # compute positional encoding for different pixels
                L = self.config.n_freq_pos_enc
                
                # center of crop
                center = (bbox[:2] + bbox[2:]) / 2.0
                angle_x, angle_y = np.arctan2(center[0]-intrx[0,2], intrx[0,0]), np.arctan2(center[1]-intrx[1,2], intrx[1,1])
                angle = np.array([angle_x, angle_y]).astype(np.float32)
                sample['center_angle'] = angle

                # corners of crop
                corner = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[1]], [bbox[2], bbox[3]]])
                corner = np.stack([corner[:,0]-intrx[0,2], corner[:,1]-intrx[1,2]], axis=-1)
                angle = np.arctan2(corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                sample["corner_angle"] = angle

            if pos_enc is not None and pos_enc == 'dense_latent':
                # dense positional encoding for all pixels
                x_grid, y_grid = range(bbox[0], bbox[2]+1), range(bbox[1], bbox[3]+1)
                x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
                pix = np.stack([x_grid-intrx[0,2], y_grid-intrx[1,2]], axis=-1)
                angle = np.arctan2(pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                angle_fdim = np.zeros((angle.shape[0], w, h))
                angle_fdim[:angle.shape[0], :angle.shape[1], :angle.shape[2]] = angle
                sample["dense_angle"] = angle_fdim.astype(np.float32)
                mask = np.zeros((w, h))
                mask[:angle.shape[1], :angle.shape[2]] = 1
                sample["dense_mask"] = mask.astype(np.float32)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 
                  'image_path': sample_path.split()[0], 
                  'depth_path': sample_path.split()[1]}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = self.rng.randint(0, img.shape[1] - width + 1)
        y = self.rng.randint(0, img.shape[0] - height + 1)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        crop_box = np.array([x, x+width, y, y+height], dtype=np.float32)
        return img, depth, crop_box 
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = self.rng.random()
        if do_translate > p:
            return img, depth
        x = self.rng.randint(-max_t, max_t+1)
        y = self.rng.randint(-max_t, max_t+1)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt, bbox, size=(640,480), use_flip_aug=False, use_scale_aug=False):
        new_bbox = bbox.copy()
        flipped = False
        scaled = False
        if self.config.aug:
            # Random flipping
            if use_flip_aug:
                do_flip = self.rng.random()
                if do_flip > 0.5:
                    image = (image[:, ::-1, :]).copy()
                    depth_gt = (depth_gt[:, ::-1, :]).copy()
                    new_bbox[0] = size[0] - bbox[1]
                    new_bbox[1] = size[0] - bbox[0]
                    flipped = True

            # Random gamma, brightness, color augmentation
            do_augment = self.rng.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

            if use_scale_aug:
                do_scale = self.rng.random()
                if do_scale > 0.5:
                    scale_factor = self.rng.uniform(0.8, 1.2)
                    new_h = int(image.shape[0] * scale_factor)
                    new_w = int(image.shape[1] * scale_factor)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR).clip(0, 1)
                    depth_gt = cv2.resize(depth_gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    # take center crop of size (512, 384)
                    top_margin = int((new_h - 384) / 2)
                    left_margin = int((new_w - 512) / 2)
                    image = image[top_margin:top_margin + 384, left_margin:left_margin + 512, :]
                    depth_gt = depth_gt[top_margin:top_margin + 384, left_margin:left_margin + 512].reshape((384,512,1))
                    new_bbox = (new_bbox * 0.8) / scale_factor # fixed size (384, 512) input to the network
                    scaled = True

        return image, depth_gt, new_bbox, flipped, scaled

    def augment_image(self, image):
        # gamma augmentation
        gamma = self.rng.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = self.rng.uniform(0.75, 1.25)
        else:
            brightness = self.rng.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = self.rng.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        if self.mode == 'online_eval':
            return len(self.filenames) # Will test on 3 random crops, multiple by some factor (3 or 5)
        else:
            return len(self.filenames)


class EvalDataset(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        
        online_eval_file_name = config.filenames_file_eval
        with open(online_eval_file_name, 'r') as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()
        self.reset_rng()

    def reset_rng(self, seed=0):
        # print(f'Reseting RNG with seed={seed}') 
        self.rng = np.random.RandomState(seed)

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        idx = idx % len(self.filenames) # this is useful when running evaluation on multiple crops
        
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}
        
        flipped = False
    
        data_path = self.config.data_path_eval

        image_path = os.path.join(
            data_path, remove_leading_slash(sample_path.split()[0]))
        image = np.asarray(self.reader.open(image_path),
                            dtype=np.float32) / 255.0
        h, w = image.shape[:2]

        gt_path = self.config.gt_path_eval
        depth_path = os.path.join(
            gt_path, remove_leading_slash(sample_path.split()[1]))
        has_valid_depth = False
        try:
            depth_gt = self.reader.open(depth_path)
            has_valid_depth = True
        except IOError:
            depth_gt = False
            # print('Missing gt for {}'.format(image_path))
        # print (image_path)

        if has_valid_depth:
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            if self.config.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            mask = np.logical_and(
                depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
        else:
            mask = False

        if self.config.do_kb_crop:
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin:top_margin + 352,
                            left_margin:left_margin + 1216, :]
            if self.mode == 'online_eval' and has_valid_depth:
                depth_gt = depth_gt[top_margin:top_margin +
                                    352, left_margin:left_margin + 1216, :]
        
        crop_box = np.array([0, 640, 0, 480], dtype=np.float32)
        if self.config.test_on_crop and self.config.aug and self.config.random_crop:
            image, depth_gt, crop_box = self.random_crop(
                image, depth_gt, self.config.input_height, self.config.input_width)

        if self.config.get('use_scale_aug', False): # check this sometime later
            # take center crop of size (512, 384)
            top_margin = int((image.shape[0] - 384) / 2)
            left_margin = int((image.shape[1] - 512) / 2)
            image = image[top_margin:top_margin + 384, left_margin:left_margin + 512, :]
            depth_gt = depth_gt[top_margin:top_margin + 384, left_margin:left_margin + 512].reshape((384,512,1))
            crop_box = (crop_box * 0.8) # fixed size (384, 512) input to the network

        sample = {'image': image, 
                    'depth': depth_gt, 
                    'focal': focal,
                    'has_valid_depth': has_valid_depth,
                    'image_path': sample_path.split()[0], 
                    'depth_path': sample_path.split()[1],
                    'mask': mask,
                    'crop_box': crop_box
                    }

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)
        
        # change crop_box to [x1, y1, x2, y2] format
        if 'crop_box' in sample:
            crop_box = sample['crop_box']
            sample['crop_box'] = np.array([crop_box[0], crop_box[2], crop_box[1], crop_box[3]], dtype=np.float32)
            bbox = sample['crop_box'].astype(np.int16)

            # taken from toolbox/camera_params.m of NYU dataset
            intrx = np.eye(3)
            intrx[0,0] = focal
            intrx[1,1] = focal
            if self.config.get('use_gt_principal', False):
                intrx[0,2] = 325.58 # check if this helps
                intrx[1,2] = 253.73 # check if this helps
                if flipped:
                    intrx[0,2] = 640 - intrx[0,2] # doesn't make much of a difference

            pos_enc = self.config.get('pos_enc', None)
            if pos_enc is not None and pos_enc == 'center+corner_latent': # compute positional encoding for different pixels
                L = self.config.n_freq_pos_enc
                
                # center of crop
                center = (bbox[:2] + bbox[2:]) / 2.0
                angle_x, angle_y = np.arctan2(center[0]-intrx[0,2], intrx[0,0]), np.arctan2(center[1]-intrx[1,2], intrx[1,1])
                angle = np.array([angle_x, angle_y]).astype(np.float32)
                sample['center_angle'] = angle

                # corners of crop
                corner = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[1]], [bbox[2], bbox[3]]])
                corner = np.stack([corner[:,0]-intrx[0,2], corner[:,1]-intrx[1,2]], axis=-1)
                angle = np.arctan2(corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                sample["corner_angle"] = angle

            if pos_enc is not None and pos_enc == 'dense_latent':
                # dense positional encoding for all pixels
                x_grid, y_grid = range(bbox[0], bbox[2]+1), range(bbox[1], bbox[3]+1)
                x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
                pix = np.stack([x_grid-intrx[0,2], y_grid-intrx[1,2]], axis=-1)
                angle = np.arctan2(pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                angle_fdim = np.zeros((angle.shape[0], w, h))
                angle_fdim[:angle.shape[0], :angle.shape[1], :angle.shape[2]] = angle
                sample["dense_angle"] = angle_fdim.astype(np.float32)
                mask = np.zeros((w, h))
                mask[:angle.shape[1], :angle.shape[2]] = 1
                sample["dense_mask"] = mask.astype(np.float32)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 
                  'image_path': sample_path.split()[0], 
                  'depth_path': sample_path.split()[1]}

        return sample

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = self.rng.randint(0, img.shape[1] - width + 1)
        y = self.rng.randint(0, img.shape[0] - height + 1)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        crop_box = np.array([x, x+width, y, y+height], dtype=np.float32)
        return img, depth, crop_box 

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, do_normalize=False, size=None):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        # print ('check', type(pic), pic.shape)
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
