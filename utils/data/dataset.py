import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from albumentations import Compose, VerticalFlip, RandomRotate90, HorizontalFlip
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms

# from utils.config import DATASET_ROOT
from . import DATASET_ROOT, TEST_IMAGES_DIR, DATA_PATH_DICT, IDS
from utils.data.augmentation import img_load, img_mask_crop
from utils.data.preprocess import IMG, GT
from .. import img_basename

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class AgricultureDataset(Dataset):
    def __init__(
            self,
            mode="train",
            filepath_dict=None,
            win_size=(256, 256),  #
            num_samples=10000,
            pre_norm=False,
            scale=1.0 / 1.0,
    ):
        """Class serving as the main interface for data interactions

        .. code-block:: text

            .
            ├── test
            │   ├── boundaries
            │   ├── images
            │   │   ├── nir
            │   │   └── rgb
            │   └── masks
            ├── train
            │   ├── boundaries
            │   ├── gt          # generated using preprocess.py
            │   ├── images
            │   │   ├── nir
            │   │   └── rgb
            │   ├── labels
            │   │   ├── double_plant
            │   │   ├── drydown
            │   │   ├── endrow
            │   │   ├── nutrient_deficiency
            │   │   ├── planter_skip
            │   │   ├── storm_damage
            │   │   ├── water
            │   │   ├── waterway
            │   │   └── weed_cluster
            │   └── masks
            └── val
                ├── boundaries
                ├── gt              # generated using preprocess.py
                ├── images
                │   ├── nir
                │   └── rgb
                ├── labels
                │   ├── double_plant
                │   ├── drydown
                │   ├── endrow
                │   ├── nutrient_deficiency
                │   ├── planter_skip
                │   ├── storm_damage
                │   ├── water
                │   ├── waterway
                │   └── weed_cluster
                └── masks

        :param mode:
        :param filepath_dict:
        :param win_size:
        :param num_samples:
        :param pre_norm:
        :param scale:
        """
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.norm = pre_norm
        self.win_size = win_size
        self.samples = num_samples
        self.scale = scale
        self.all_ids = filepath_dict["all_files"]
        self.image_files = filepath_dict[IMG]  # image_files = [[bands1, bands2,..], ...]
        self.mask_files = filepath_dict[GT]  # mask_files = [gt1, gt2, ...]

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):

        if len(self.image_files) > 1:
            # collect NIR and RGB to form 4D input image
            imgs = []
            for k in range(len(self.image_files[idx])):
                filename = self.image_files[idx][k]
                path, _ = os.path.split(filename)
                # expand and aggregate nir images
                if path[-3:] == "nir":
                    img = img_load(filename, gray=True, scale_rate=self.scale)
                    img = np.expand_dims(img, 2)
                    imgs.append(img)
                # aggregate rgb
                else:
                    img = img_load(filename, scale_rate=self.scale)
                    imgs.append(img)
            image = np.concatenate(imgs, 2)
        else:
            filename = self.image_files[idx][0]
            path, _ = os.path.split(filename)
            if path[-3:] == "nir":
                image = img_load(filename, gray=True, scale_rate=self.scale)
                image = np.expand_dims(image, 2)  # (H,W) -> (H,W,C)
            else:
                image = img_load(filename, scale_rate=self.scale)

        label = img_load(self.mask_files[idx], gray=True, scale_rate=self.scale)

        if self.win_size != label.shape:
            image, label = img_mask_crop(
                image=image, mask=label, size=self.win_size, limits=self.win_size
            )

        # process
        if self.mode == "train":
            image_p, label_p = self.train_augmentation(image, label)
        elif self.mode == "val":
            image_p, label_p = self.val_augmentation(image, label)

        # preprocess with numpy tranposing of array to get (512,512,4)
        image_p = np.asarray(image_p, np.float32).transpose((2, 0, 1)) / 255.0
        label_p = np.asarray(label_p, dtype="int64")

        image_p, label_p = torch.from_numpy(image_p), torch.from_numpy(label_p)

        if self.norm:
            image_p = self.normalize(image_p)

        return image_p, label_p

    @classmethod
    def train_augmentation(cls, img, mask):

        return AgricultureDataset.data_augmentation(img, mask, horizontal_flip=0.5, vertical_flip=0.5,
                                                    random_rotate_90=0.5)
        # augment = Compose(
        #     [
        #         VerticalFlip(p=0.5),
        #         HorizontalFlip(p=0.5),
        #         RandomRotate90(p=0.5),
        #         # MedianBlur(p=0.2),
        #         # Transpose(p=0.5),
        #         # RandomSizedCrop(min_max_height=(128, 512), height=512, width=512, p=0.1),
        #         # ShiftScaleRotate(p=0.2,
        #         #                  rotate_limit=10, scale_limit=0.1),
        #         # ChannelShuffle(p=0.1),
        #     ]
        # )
        #
        # augmented = augment(image=img, mask=mask)
        # return augmented["image"], augmented["mask"]

    @classmethod
    def val_augmentation(cls, img, mask):
        return AgricultureDataset.data_augmentation(img, mask, horizontal_flip=0.5, vertical_flip=0.5,
                                                    random_rotate_90=0.5)
        # aug = Compose(
        #     [VerticalFlip(p=0.5), HorizontalFlip(p=0.5), RandomRotate90(p=0.5), ]
        # )
        #
        # augmented = aug(image=img, mask=mask)
        # return augmented["image"], augmented["mask"]

    @classmethod
    def data_augmentation(cls, img, mask, vertical_flip: float = 0.5,
                          horizontal_flip: float = 0.5,
                          random_rotate_90: float = 0.5) -> Tuple:
        augment = Compose(
            [
                VerticalFlip(p=vertical_flip),
                HorizontalFlip(p=horizontal_flip),
                RandomRotate90(p=random_rotate_90),
                # MedianBlur(p=0.2),
                # Transpose(p=0.5),
                # RandomSizedCrop(min_max_height=(128, 512), height=512, width=512, p=0.1),
                # ShiftScaleRotate(p=0.2,
                #                  rotate_limit=10, scale_limit=0.1),
                # ChannelShuffle(p=0.1),
            ]
        )
        augmented = augment(image=img, mask=mask)
        return augmented["image"], augmented["mask"]

    @classmethod
    def normalize(cls, img):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        norm = standard_transforms.Compose([standard_transforms.Normalize(*mean_std)])
        return norm(img)


class AgricultureDataset2021(AgricultureDataset):
    def __init__(self):
        super().__init__()
        palette_land = {
            0: (0, 0, 0,),  # TODO: unsure if relabel is necessary leaving as `background`
            1: (255, 255, 0),  # TODO: UPDATED [1] to `water` from `cloud_shadow`
            2: (255, 0, 255),  # double_plant
            3: (0, 255, 0),  # planter_skip
            4: (0, 0, 255),  # TODO: UPDATED [2] to `drydown` from `standing_water`
            5: (255, 255, 255),  # waterway
            6: (0, 255, 255),  # weed_cluster
            7: (0, 128, 255),  # TODO: UPDATED [3] to `endrow` from `cloud_shadow`
            8: (128, 0, 128,),  # TODO: UPDATED [4] to `nutrient_deficiency` from `cloud_shadow`
            9: (255, 0, 0),  # TODO: UPDATED [5] to `storm_damage` from `cloud_shadow`
        }

        # customised palette for visualization, easier for reading in paper
        palette_vsl = {
            0: (
                0,
                0,
                0,
            ),  # TODO:  not sure whether to update or not `background`, labels reflected only 6 / 7
            1: (0, 255, 0),  # UPDATED [1] to `water`
            2: (255, 0, 0),  # double_plant
            3: (0, 200, 200),  # planter_skip
            4: (255, 255, 255),  # UPDATED [2] to `drydown`
            5: (128, 128, 0),  # waterway
            6: (0, 0, 255),  # weed_cluster
            7: (0, 128, 255),  # UPDATED [3] to `endrow`
            8: (128, 0, 128),  # UPDATED [4] to `nutrient_deficiency`
            9: (128, 255, 128),  # UPDATED [5] to `storm_damage`
        }

        labels_folder = {
            "water": 1,  # TODO: UPDATED [1] to `water` from `cloud_shadow`
            "double_plant": 2,
            "planter_skip": 3,
            "drydown": 4,  # TODO: UPDATED [2] to `drydown` from `standing_water`
            "waterway": 5,
            "weed_cluster": 6,
            "endrow": 7,
            "nutrient_deficiency": 8,
            "storm_damage": 9,
        }

        # TODO: likely needs to be updated to reflect the new 9 classes -- include background...?
        # OLD
        # land_classes = ["background", "cloud_shadow", "double_plant", "planter_skip",
        #                 "standing_water", "waterway", "weed_cluster"]

        # UPDATED labels: only 6 classes
        land_classes = [
            "background",
            "water",
            "double_plant",
            "planter_skip",
            "drydown",
            "waterway",
            "weed_cluster",
            "endrow",
            "nutrient_deficiency",
            "storm_damage",
        ]
        Data_Folder = {
            "Agriculture": {
                "ROOT": DATASET_ROOT,
                "RGB": "images/rgb/{}.jpg",
                "NIR": "images/nir/{}.jpg",
                "SHAPE": (512, 512),
                "GT": "gt/{}.png",
            },
        }

        IMG = "images"  # RGB or IRRG, rgb/nir
        GT = "gt"
        IDS = "IDs"


def get_input_entry(
        data_directory: [
            Path or str] = "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test/",
        entry_id: str = "ZTUK7EGVR_9431-3663-9943-4175") -> dict:
    """
    **NOTE** Access Keys:
        - boundaries
        - nir
        - rgb
        - masks

    :param data_directory:
    :param entry_id:
    :return:
    """
    # load from file
    SUB_DIRS = ["boundaries", 'images/nir', 'images/rgb', 'masks']
    TEST_ROOT = Path(
        data_directory
    )
    ID = entry_id
    input_entry = {
    }
    for sub_dir in SUB_DIRS:
        try:
            path = TEST_ROOT / sub_dir / (ID + ".*")
            files = glob.glob(str(path))
            if "nir" in sub_dir:
                input_entry["nir"] = files[0]
            elif "rgb" in sub_dir:
                input_entry["rgb"] = files[0]
            else:
                input_entry[sub_dir] = files[0]
            # print(glob.glob(str(path))) # DEBUG
        except Exception as e:
            print(e)
            pass
    return input_entry


def get_real_test_list(
        root_folder=TEST_IMAGES_DIR, data_folder=DATA_PATH_DICT, name="Agriculture", bands=["RGB"]
):
    dict_list = {}
    basename = [img_basename(f) for f in os.listdir(os.path.join(root_folder, "nir"))]
    dict_list["all"] = basename

    test_dict = {
        IDS: dict_list,
        IMG: [
            os.path.join(data_folder[name]["ROOT"], "test", data_folder[name][band])
            for band in bands
        ],
        # GT: os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT'])
        # IMG: [[img_id.format(id) for img_id in img_ids] for id in test_ids]
    }
    return test_dict