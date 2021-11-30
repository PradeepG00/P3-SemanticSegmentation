import os

# from ..config import DATASET_ROOT

DATASET_ROOT = "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021"
# change DATASET ROOT to your dataset path
# DATASET_ROOT = "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021"

TEST_DIR = os.path.join(DATASET_ROOT, "test")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "test/images")
