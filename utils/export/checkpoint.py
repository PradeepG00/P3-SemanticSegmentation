# from lib.core.mscgnet import *

# score 0.547, no TTA
import datetime
import logging
import os

import numpy as np
import torch
from PIL import Image

# from core.net import *
from core.net import get_model
from utils.data.augmentation import img_load
from utils.data.preprocess import IDS, GT, IMG
from utils import PROJECT_ROOT

#####################################
# Setup Logging
# TODO: need to eventually cut after refacotirzation
#####################################
from utils.gpu import get_available_gpus

model_name = "ensemble"
output_path = PROJECT_ROOT / "submission" / "results_checkpoint1_checkpoint2_tta"
logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
rootLogger = logging.getLogger()

log_path = PROJECT_ROOT / "logs/{0}/{1}.log".format(
    f"/{model_name}", f"{model_name}-{datetime.datetime.now():%d-%b-%y-%H:%M:%S}"
)
log_dir = PROJECT_ROOT / f"logs/{model_name}"
if os.path.exists(log_dir):
    print("Saving log files to:", log_dir)
else:
    print("Creating log directory:", log_dir)
    os.mkdir(log_dir)

fileHandler = logging.FileHandler(log_path)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

checkpoint1 = {
    "core": "MSCG-Rx50",
    "data": "Agriculture",
    "bands": ["NIR", "RGB"],
    "num_classes": 10,
    "nodes": (32, 32),
    "snapshot": "checkpoints/adam/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/MSCG-Rx50-epoch_6_loss_1.09903_acc_0.77739_acc-cls_0.53071_mean-iu_0.39789_fwavacc_0.64861_f1_0.53678_lr_0.0000845121.pth",
}

# score 0.550 , no TTA
checkpoint2 = {
    "core": "MSCG-Rx101",
    "data": "Agriculture",
    "bands": ["NIR", "RGB"],
    "num_classes": 10,
    "nodes": (32, 32),
    "snapshot": "checkpoints/adam/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_4_loss_1.26896_acc_0.77713_acc-cls_0.54260_mean-iu_0.40996_fwavacc_0.64399_f1_0.55334_lr_0.0001245001.pth",
}

checkpoint3 = {
    "core": "MSCG-Rx101",
    "data": "Agriculture",
    "bands": ["NIR", "RGB"],
    "num_classes": 10,
    "nodes": (32, 32),
    "snapshot": "checkpoints/epoch_15_loss_0.88412_acc_0.88690_acc-cls_0.78581_"
                "mean-iu_0.68205_fwavacc_0.80197_f1_0.80401_lr_0.0001075701.pth",
}


# checkpoint1 + checkpoint2, test score 0.599,
# checkpoint1 + checkpoint2 + checkpoint3, test score 0.608


def get_net(checkpoint_path: str, checkpoint=checkpoint1, use_gpu: bool = True):
    """Function for loading an MSCG-Net from a .pth checkpoint file path

    :param checkpoint: Dictionary containing model meta-data for loading from a .pth
    :return:
    """
    net = get_model(
        name=checkpoint["core"],
        classes=checkpoint["num_classes"],
        node_size=checkpoint["nodes"],
    )
    print(checkpoint["num_classes"],
        checkpoint["nodes"])

    checkpoint_path = PROJECT_ROOT / (checkpoint["snapshot"].replace("../", ""))
    logging.debug("Loading from {}".format(str(checkpoint_path)))
    print(get_available_gpus(6, "gb"))
    available_gpus = get_available_gpus(6, "gb")
    # if len(available_gpus) == 0:
    net.load_state_dict(torch.load(str(checkpoint_path), map_location=torch.device("cpu")))
    # else:
    #     net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(available_gpus[0])))
    if use_gpu:
        net.cuda()
    else:
        net.cpu()
    net.eval()
    return net


def load_test_img(test_files):
    """Load the set of test images
    TODO: what is data struct of test_files?

    :param test_files:
    :return:
    """
    id_dict = test_files[IDS]
    image_files = test_files[IMG]
    # mask_files = test_files[GT]

    for key in id_dict.keys():
        for id in id_dict[key]:
            if len(image_files) > 1:
                imgs = []
                for i in range(len(image_files)):
                    filename = image_files[i].format(id)
                    path, _ = os.path.split(filename)
                    if path[-3:] == "nir":
                        # img = imload(filename, gray=True)
                        img = np.asarray(Image.open(filename), dtype="uint8")
                        img = np.expand_dims(img, 2)

                        imgs.append(img)
                    else:
                        img = img_load(filename)
                        imgs.append(img)
                image = np.concatenate(imgs, 2)
            else:
                filename = image_files[0].format(id)
                path, _ = os.path.split(filename)
                if path[-3:] == "nir":
                    # image = imload(filename, gray=True)
                    image = np.asarray(Image.open(filename), dtype="uint8")
                    image = np.expand_dims(image, 2)
                else:
                    image = img_load(filename)
            # label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')

            yield image


def load_ids(test_files):
    """Generator function for loading the test set of images from disk

    :param test_files:
    :return:
    """
    id_dict = test_files[IDS]

    for key in id_dict.keys():
        for id in id_dict[key]:
            yield id


def load_gt(test_files):
    """Generator function for loading the test set of ground-truth files from disk
    TODO: what is the type/datastruct of `test_files`?

    :param test_files:
    :return:
    """
    id_dict = test_files[IDS]
    mask_files = test_files[GT]
    for key in id_dict.keys():
        for id in id_dict[key]:
            label = np.asarray(Image.open(mask_files.format(id)), dtype="uint8")
            yield label
