import os

# from ..config import DATASET_ROOT

# change DATASET ROOT to your dataset path
# DATASET_ROOT = "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021"


# Directory routing
DATASET_ROOT = "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021"
TEST_DIR = os.path.join(DATASET_ROOT, "test")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "test/images")
TEST_IM_RGB_DIR = os.path.join(DATASET_ROOT, "test/images/rgb")
TEST_IM_NIR_DIR = os.path.join(DATASET_ROOT, "test/images/nir")


# Visualization consants
PALETTE_LAND = {
    0: (0, 0, 0),  # `background`
    1: (255, 255, 0),  # `water`
    2: (255, 0, 255),  # double_plant
    3: (0, 255, 0),  # planter_skip
    4: (0, 0, 255),  # `drydown`
    5: (255, 255, 255),  # waterway
    6: (0, 255, 255),  # weed_cluster
    7: (0, 128, 255),  # `endrow`
    8: (128, 0, 128),  # `nutrient_deficiency`
    9: (255, 0, 0),  # `storm_damage`
}
"""RGB Format for coloring when writing to file
"""

# customised palette for visualization, easier for reading in paper
COLOR_PALETTE = {
    # 0: (0, 0, 0,),  # `background`
    # 1: (0, 255, 0),  # `water`
    # 2: (255, 0, 0),  # double_plant
    # 3: (0, 200, 200),  # planter_skip
    # 4: (255, 255, 255),  # `drydown`
    # 5: (128, 128, 0),  # waterway
    # 6: (0, 0, 255),  # weed_cluster
    # 7: (0, 128, 255),  # `endrow`
    # 8: (128, 0, 128),  # `nutrient_deficiency`
    # 9: (128, 255, 128),  # `storm_damage`

    0: (0, 0, 0),  # `background`
    1: (255, 255, 0),  # `water`
    2: (255, 0, 255),  # double_plant
    3: (0, 255, 0),  # planter_skip
    4: (0, 0, 255),  # `drydown`
    5: (255, 255, 255),  # waterway
    6: (0, 255, 255),  # weed_cluster
    7: (0, 128, 255),  # `endrow
    8: (128, 0, 128),  # `nutrient_deficiency`
    9: (255, 0, 0),  # `storm_damage`
}
"""RGB Format for coloring when writing to file
"""

# Mapping of label folders to palette
LABELS_FOLDER = {
    "water": 1,
    "double_plant": 2,
    "planter_skip": 3,
    "drydown": 4,
    "waterway": 5,
    "weed_cluster": 6,
    "endrow": 7,
    "nutrient_deficiency": 8,
    "storm_damage": 9,
}

# OLD
# land_classes = ["background", "cloud_shadow", "double_plant", "planter_skip",
#                 "standing_water", "waterway", "weed_cluster"]

# UPDATED labels for CV for Agriculture 2021
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

DATA_PATH_DICT = {
    "Agriculture": {
        "ROOT": DATASET_ROOT,
        "RGB": "images/rgb/{}.jpg",
        "NIR": "images/nir/{}.jpg",
        "SHAPE": (512, 512),
        "GT": "gt/{}.png",
    },
    # TODO: add a .csv mapping it all out
    # TODO: add a .txt file of all the classes
}
"""Dictionary constant defining how to interface with the dataset structure
    .. code-block: text
        {
            "Agriculture": {
                    "ROOT": DATASET_ROOT,
                    
                    "RGB": "images/rgb/{}.jpg",
                    
                    "NIR": "images/nir/{}.jpg",
                    
                    "SHAPE": (512, 512),
                    
                    "GT": "gt/{}.png",
                },
        }
"""

IMG = "images"  # RGB or IRRG, rgb/nir
GT = "gt"
IDS = "IDs"
