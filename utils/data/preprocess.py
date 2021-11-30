from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split, KFold

import cv2

from utils import check_mkdir
from utils.data import TRAIN_DIR, VAL_DIR, TEST_IMAGES_DIR
from utils.data import DATASET_ROOT
# from utils.data.dataset import DATASET_ROOT

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
PALETTE_LAND = {
    0: (0, 0, 0),  # TODO: unsure if relabel is necessary leaving as `background`
    1: (255, 255, 0),  # TODO: UPDATED [1] to `water` from `cloud_shadow`
    2: (255, 0, 255),  # double_plant
    3: (0, 255, 0),  # planter_skip
    4: (0, 0, 255),  # TODO: UPDATED [2] to `drydown` from `standing_water`
    5: (255, 255, 255),  # waterway
    6: (0, 255, 255),  # weed_cluster
    7: (0, 128, 255),  # TODO: UPDATED [3] to `endrow` from `cloud_shadow`
    8: (128, 0, 128),  # TODO: UPDATED [4] to `nutrient_deficiency` from `cloud_shadow`
    9: (255, 0, 0),  # TODO: UPDATED [5] to `storm_damage` from `cloud_shadow`
}

# customised palette for visualization, easier for reading in paper
PALETTE_VIZ = {
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

LABELS_FOLDER = {
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
LAND_CLASSES = [
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
METADATA_DIR_DICT = {
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


# def check_mkdir(dir_name):
#     if not os.path.exists(dir_name):
#         os.mkdir(dir_name)


def img_basename(filename) -> str:
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename) -> bool:
    return any(filename.endswith(ext) for ext in [".png", ".jpg"])


def prepare_gt(root_folder=TRAIN_DIR, out_path="gt") -> None:
    """Function for creating ground-truths from a specified directory


    .. code-block:: text

        # pre-process
        Agriculture-Vision
        |-- train       # source directory
        |   |-- masks
        |   |-- labels
        |   |-- boundaries
        |   |-- images
        |   |   |-- nir
        |   |   |-- rgb
        |   |-- gt      # resulting output directory
        .   .
        .   .
        .   .
        |-- test
        |   |-- boundaries
        |   |-- images
        |   |   |-- nir
        |   |   |-- rgb
        |   |-- masks
        .
        .
        .

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


def reset_gt(root_folder=TRAIN_DIR, out_path="gt"):
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


def get_training_list(root_folder=TRAIN_DIR, count_label=True):
    dict_list = {}
    basename = [
        img_basename(f) for f in os.listdir(os.path.join(root_folder, "images/nir"))
    ]
    if count_label:
        for key in LABELS_FOLDER.keys():
            no_zero_files = []
            for fname in basename:
                gt = np.array(
                    cv2.imread(os.path.join(root_folder, "labels", key, fname + ".png"), -1)
                )
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basename
    # print(len(list[key]), list[key][0:5])


def split_train_val_test_sets(
        data_folder=METADATA_DIR_DICT,
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
    train_id, t_list = get_training_list(root_folder=TRAIN_DIR, count_label=False)
    val_id, v_list = get_training_list(root_folder=VAL_DIR, count_label=False)

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


def get_real_test_list(
        root_folder=TEST_IMAGES_DIR, data_folder=METADATA_DIR_DICT, name="Agriculture", bands=["RGB"]
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
