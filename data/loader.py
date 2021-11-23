import config
from data.augment import imload, img_mask_crop, VerticalFlip, HorizontalFlip, RandomRotate90, Compose

import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
import inspect
import cv2
import os
import numpy as np
from data.preprocess import *

# CONFIG = config.configure()
cv2.setNumThreads(0)  # TODO: is this necessary?
cv2.ocl.setUseOpenCL(False)


class AgricultureDataset(Dataset):
    def __init__(self, mode='train', file_lists=None, window_size=(256, 256),
                 num_samples=10000, pre_norm=False, scale=1.0 / 1.0):
        logging.debug(inspect.currentframe().f_code.co_name) # DEBUG # if CONFIG["TRACING_ENABLED"]: logging.debug(inspect.currentframe().f_code.co_name) # DEBUG

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.norm = pre_norm
        self.window_size = window_size
        self.samples = num_samples
        self.scale = scale
        self.all_ids = file_lists['all_files']
        self.image_files = file_lists[IMG]  # image_files = [[bands1, bands2,..], ...]
        self.mask_files = file_lists[GT]  # mask_files = [gt1, gt2, ...]

    def __len__(self):
        logging.debug(inspect.currentframe().f_code.co_name) # DEBUG # if CONFIG["TRACING_ENABLED"]: logging.debug(inspect.currentframe().f_code.co_name) # DEBUG

        return len(self.all_ids)

    def __getitem__(self, idx):
        logging.debug(inspect.currentframe().f_code.co_name) # DEBUG # if CONFIG["TRACING_ENABLED"]: logging.debug(inspect.currentframe().f_code.co_name) # DEBUG

        if len(self.image_files) > 1:
            images = []
            for k in range(len(self.image_files[idx])):
                filename = self.image_files[idx][k]
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    img = imload(filename, gray=True, scale_rate=self.scale)
                    img = np.expand_dims(img, 2)
                    images.append(img)
                else:
                    img = imload(filename, scale_rate=self.scale)
                    images.append(img)
            image = np.concatenate(images, 2)
        else:
            filename = self.image_files[idx][0]
            path, _ = os.path.split(filename)
            if path[-3:] == 'nir':
                image = imload(filename, gray=True, scale_rate=self.scale)
                image = np.expand_dims(image, 2)
            else:
                image = imload(filename, scale_rate=self.scale)
        # print()
        label = imload(self.mask_files[idx], gray=True, scale_rate=self.scale)

        if self.window_size != label.shape:
            image, label = img_mask_crop(image=image, mask=label,
                                         size=self.window_size, limits=self.window_size)

        if self.mode == 'train':
            image_p, label_p = self.train_augmentation(image, label)
        elif self.mode == 'val':
            image_p, label_p = self.val_augmentation(image, label)

        image_p = np.asarray(image_p, np.float32).transpose((2, 0, 1)) / 255.0
        label_p = np.asarray(label_p, dtype='int64')

        image_p, label_p = torch.from_numpy(image_p), torch.from_numpy(label_p)

        if self.norm:
            image_p = self.normalize(image_p)

        return image_p, label_p

    @classmethod
    def train_augmentation(cls, img, mask):
        logging.debug(inspect.currentframe().f_code.co_name) # DEBUG # if CONFIG["TRACING_ENABLED"]: logging.debug(inspect.currentframe().f_code.co_name) # DEBUG
        augment = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            # MedianBlur(p=0.2),
            # Transpose(p=0.5),
            # RandomSizedCrop(min_max_height=(128, 512), height=512, width=512, p=0.1),
            # ShiftScaleRotate(p=0.2,
            #                  rotate_limit=10, scale_limit=0.1),
            # ChannelShuffle(p=0.1),
        ])

        augmentation = augment(image=img, mask=mask)
        return augmentation['image'], augmentation['mask']

    @classmethod
    def val_augmentation(cls, img, mask):
        logging.debug(inspect.currentframe().f_code.co_name) # DEBUG # if CONFIG["TRACING_ENABLED"]: logging.debug(inspect.currentframe().f_code.co_name) # DEBUG
        augment = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
        ])

        augmentation = augment(image=img, mask=mask)
        return augmentation['image'], augmentation['mask']

    @classmethod
    def normalize(cls, img):
        logging.debug(inspect.currentframe().f_code.co_name) # DEBUG # if CONFIG["TRACING_ENABLED"]: logging.debug(inspect.currentframe().f_code.co_name) # DEBUG
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        normalize = standard_transforms.Compose([standard_transforms.Normalize(*mean_std)])
        return normalize(img)
