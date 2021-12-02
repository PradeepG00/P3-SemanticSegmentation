from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import logging
import os
from pathlib import Path
from turtle import st
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray
from sklearn.model_selection import KFold

from utils import check_mkdir, img_basename
from utils.data import TRAIN_DIR, VAL_DIR, TEST_IMAGES_DIR, DATA_PATH_DICT, IDS, GT, LABELS_FOLDER, IMG
from utils.data.augmentation import img_load

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

"""
In the loaded numpy array, only 0-6 integer labels are allowed, and they represent the annotations in the following way:

0 - background
1 - cloud_shadow
2 - double_plant
3 - planter_skip
4 - standing_water
5 - waterway
6 - weed_cluster

"""


# def check_mkdir(dir_name):
#     if not os.path.exists(dir_name):
#         os.mkdir(dir_name)


# def img_basename(filename) -> str:
#     return os.path.basename(os.path.splitext(filename)[0])
#
#
# def is_image(filename) -> bool:
#     return any(filename.endswith(ext) for ext in [".png", ".jpg"])


def get_processed_masks_and_boundaries(mask_path: str, boundary_path: str) -> tuple[ndarray, ndarray]:
    """Utility function for loading and applying preprocessing to mask and boundary image array
    :param paths:
    :return: tuple[ndarray, ndarray]
    """
    masks = np.array(
        cv2.imread(
            mask_path, -1, ) / 255,
        dtype=int,
    )
    boundaries = np.array(
        cv2.imread(
            boundary_path, -1, ) / 255,
        dtype=int,
    )
    return masks, boundaries


def prepare_ground_truth(root_folder=TRAIN_DIR, out_path="gt") -> None:
    """Function for creating ground-truths from a specified directory


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

    :param root_folder:
    :param out_path:
    :return:
    """
    if not os.path.exists(os.path.join(root_folder, out_path)):
        print("----------creating groundtruth data for training./.val---------------")
        check_mkdir(os.path.join(root_folder, out_path))
        basname = [
            img_basename(f) for f in os.listdir(os.path.join(root_folder, "images/rgb"))
        ]
        gt = basname[0] + ".png"
        # this is something that could be multiprocessed
        for fname in basname:
            gtz = np.zeros((512, 512), dtype=int)
            for key in LABELS_FOLDER.keys():
                gt = fname + ".png"
                mask = (
                        np.array(
                            cv2.imread(os.path.join(root_folder, "labels", key, gt), -1)
                            / 255,
                            dtype=int,
                        )
                        * LABELS_FOLDER[key]
                )
                gtz[gtz < 1] = mask[gtz < 1]

            for key in ["boundaries", "masks"]:
                mask = np.array(
                    cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int
                )
                gtz[mask == 0] = 255
            logging.debug(
                "writing to... {}".format(os.path.join(root_folder, out_path, gt))
            )
            cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)


def reset_ground_truth(root_folder=TRAIN_DIR, out_path="gt"):
    """***BREAKING***

    :param root_folder:
    :param out_path:
    :return:
    """
    return
    # if not os.path.exists(os.path.join(root_folder, out_path)):
    #     print("----------deleting groundtruth data for training./.val---------------")
    #     # check_mkdir(os.path.join(root_folder, out_path))
    #     basename = [
    #         img_basename(f) for f in os.listdir(os.path.join(root_folder, "images/rgb"))
    #     ]
    #     gt = basename[0] + ".png"
    #     # this is something that could be multiprocessed
    #     # for fname in basname:
    #     #     gtz = np.zeros((512, 512), dtype=int)
    #     #     for key in labels_folder.keys():
    #     #         gt = fname + '.png'
    #     #         mask = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, gt), -1) / 255, dtype=int) * \
    #     #                labels_folder[key]
    #     #         gtz[gtz < 1] = mask[gtz < 1]
    #     #
    #     #     for key in ['boundaries', 'masks']:
    #     #         mask = np.array(cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int)
    #     #         gtz[mask == 0] = 255
    #     #     logging.debug("writing to... {}".format(os.path.join(root_folder, out_path, gt)))
    #     os.remove(os.path.join(root_folder, out_path, gt))
    #     # cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)


def get_training_list(source=TRAIN_DIR, count_label=True):
    """

    :param source: directory to containing subdirectories of data
    :param count_label:
    :return:
    """
    dict_list = {}
    basename = [
        img_basename(f) for f in os.listdir(os.path.join(source, "images/nir"))
    ]
    if count_label:
        for key in LABELS_FOLDER.keys():
            no_zero_files = []
            for fname in basename:
                gt = np.array(
                    cv2.imread(os.path.join(source, "labels", key, fname + ".png"), -1)
                )
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basename
    # print(len(list[key]), list[key][0:5])


def split_train_val_test_sets(
        data_folder=DATA_PATH_DICT,
        name="Agriculture",
        bands=["NIR", "RGB"],
        KF=3,
        k=1,
        seeds=69278,
) -> Tuple[dict]:
    """

    :param data_folder:
    :param name:
    :param bands:
    :param KF:
    :param k:
    :param seeds:
    :return:
    """
    train_id, t_list = get_training_list(source=TRAIN_DIR, count_label=False)
    val_id, v_list = get_training_list(source=VAL_DIR, count_label=False)

    # create k-folds dataset of folder paths
    if KF >= 2:
        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)
        val_ids = np.array(v_list)
        idx = list(kf.split(np.array(val_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0
        t2_list, v_list = list(val_ids[idx[k][0]]), list(val_ids[idx[k][1]])
    else:
        t2_list = []

    img_folders = [
        os.path.join(data_folder[name]["ROOT"], "train", data_folder[name][band])
        for band in bands
    ]
    gt_folder = os.path.join(
        data_folder[name]["ROOT"], "train", data_folder[name]["GT"]
    )

    val_folders = [
        os.path.join(data_folder[name]["ROOT"], "val", data_folder[name][band])
        for band in bands
    ]
    val_gt_folder = os.path.join(
        data_folder[name]["ROOT"], "val", data_folder[name]["GT"]
    )
    #                    {}
    train_dict = {
        IDS: train_id,
        IMG: [[img_folder.format(id) for img_folder in img_folders] for id in t_list]
             + [[val_folder.format(id) for val_folder in val_folders] for id in t2_list],
        GT: [gt_folder.format(id) for id in t_list]
            + [val_gt_folder.format(id) for id in t2_list],
        "all_files": t_list + t2_list,
    }

    val_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
        "all_files": v_list,
    }

    test_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
    }

    print("train set -------", len(train_dict[GT]))
    print("val set ---------", len(val_dict[GT]))
    return train_dict, val_dict, test_dict


def reshape_im_rgb_nir(im_nir, im_rgb):
    im_nir = np.expand_dims(im_nir, 2)  # expand along the 2 dimension
    print("new nir", im_nir)
    print("new nir shape", im_nir.shape)
    im_4d = [im_nir, im_rgb]
    im_4d = np.concatenate(im_4d, 2)
    print("nir + rgb", im_4d.shape)
    im_4d = np.asarray(im_4d, dtype=np.float32)
    print(im_4d.shape)
    return im_4d.copy()





def preprocess_fusion_input(in_img, norm: bool = True, DEBUG: bool = True) -> np.array:
    """

    :param in_img:
    :param norm:
    :param DEBUG:
    :return:
    """
    _img = np.asarray(in_img, dtype="float32")
    if DEBUG: print("Input NumPy Shape:", _img.shape); print("Input NumPy Shape:".format(_img.shape))  # DEBUG
    _img = st.ToTensor()(_img)
    if DEBUG: print("Input Tensor Shape:", _img.shape);
    print(
        "Input Tensor Shape: {}".format(_img.shape))  # DEBUG
    _img = _img / 255.0
    # if DEBUG:
    #     print("_img / 255 = ", img);
    # print("_img / 255 = ", img)  # DEBUG# DEBUG
    # if norm:
    #     _img = st.Normalize(*mean_std)(img)
    # print(f"normalizing image: {img}")
    _im = _img.cpu().numpy().transpose(
        (1, 2, 0))  # after applying the necessary transformations we go to 3-dims numpy 512,512,4... why?
    return _im


def preprocess_np_im_entry(entry: dict, scale: float = 1.0, DEBUG: bool = True, normalize: bool = True):
    """

    :param entry:
    :param scale:
    :param DEBUG:
    :return:
    """
    im_rgb, im_nir, mask, bound = img_load(entry["rgb"], scale_rate=scale), \
                                  img_load(entry["nir"], scale_rate=scale, gray=True), \
                                  img_load(entry["masks"], gray=True, scale_rate=scale), \
                                  img_load(entry["boundaries"], gray=True, scale_rate=scale)

    # access and properly reshape
    print("rgb", im_rgb.shape,  # H x W x C
          "nir", im_nir.shape)  # DEBUG
    # print("rgb", im_rgb)
    # print("nir", im_nir)
    # 512 x 512 or H x W
    im_nir = np.expand_dims(im_nir, 2)  # expand along the 2 dimension
    print("new nir", im_nir)
    print("new nir shape", im_nir.shape)
    im_4d = [im_nir, im_rgb]
    rv = np.concatenate(im_4d, 2)
    print("nir + rgb", rv.shape)
    rv = np.asarray(rv, dtype=np.float32)

    # im_rand = np.random.rand(512,512,1)
    # print("pre-rand", im_rand.shape)
    # im_rand = np.expand_dims(im_rand, 2)
    # print("post-rand", im_rand.shape)
    # return rv
    # _img = np.asarray(in_img, dtype="float32")
    # if DEBUG: print("Input NumPy Shape:", img.shape); print("Input NumPy Shape:".format(img.shape))  # DEBUG
    rv = st.ToTensor()(rv)
    # if DEBUG: print("Input Tensor Shape:", img.shape);
    # print(
    #     "Input Tensor Shape: {}".format(img.shape))  # DEBUG
    rv = rv / 255.0
    # # if DEBUG:
    # #     print("_img / 255 = ", img);
    # # print("_img / 255 = ", img)  # DEBUG# DEBUG
    if normalize:
        rv = st.Normalize(*MEAN_STD)(rv)
    #     # print(f"normalizing image: {img}")
    rv = rv.cpu().numpy().transpose(
        (1, 2, 0))  # after applying the necessary transformations we go to 3-dims numpy 512,512,4... why?
    print("final", rv.shape)

    return rv



