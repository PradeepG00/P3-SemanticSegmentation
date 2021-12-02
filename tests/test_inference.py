import glob
import json
import os
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as st
from PIL import Image
from torch import nn

import train_R50
from core.net import get_model, load_rx50
from utils.data.augmentations import img_load
from utils.data.visual import mean_std


def preprocess_entry(entry: dict, id: str, scale=1.0):
    im_rgb, im_nir, mask, bound = img_load(entry["rgb"], scale_rate=scale), \
                                  img_load(entry["nir"], scale_rate=scale), \
                                  img_load(entry["mask"], gray=True, scale_rate=scale), \
                                  img_load(entry["boundaries"], gray=True, scale_rate=scale)
    # gt =
    # prediction...
    gtz = np.zeros((512, 512), dtype=int)
    # mask = np.array(
    #     cv2.imread(os.path.join(root_folder, "labels", key, gt), -1)
    #     / 255,
    #     dtype=int,
    # )
    gtz[mask == 0] = 255
    # for key in LABELS_FOLDER.keys():
    #     gt = fname + ".png"
    #     mask = (
    #             np.array(
    #                 cv2.imread(os.path.join(root_folder, "labels", key, gt), -1)
    #                 / 255,
    #                 dtype=int,
    #             )
    #             * LABELS_FOLDER[key]
    #     )
    #     gtz[gtz < 1] = mask[gtz < 1]
    #
    # for key in ["boundaries", "masks"]:
    #     mask = np.array(
    #         cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int
    #     )
    #     gtz[mask == 0] = 255
    # logging.debug(
    #     "writing to... {}".format(os.path.join(root_folder, out_path, gt))
    # )
    # cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)
    # # for key in ["boundaries", "masks"]:
    # #     mask = np.array(
    # #         cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int
    # #     )
    # #     gtz[mask == 0] = 255
    # # logging.debug(
    # #     "writing to... {}".format(os.path.join(root_folder, out_path, gt))
    # # )
    # # cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)
    # pass


def get_input_entry(
        data_directory: Path or str = "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test/",
        entry_id: str = "ZTUK7EGVR_9431-3663-9943-4175") -> dict:
    """

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
        path = TEST_ROOT / sub_dir / (ID + ".*")
        files = glob.glob(str(path))
        if "nir" in sub_dir:
            input_entry["nir"] = files[0]
        elif "rgb" in sub_dir:
            input_entry["rgb"] = files[0]
        else:
            input_entry[sub_dir] = files[0]
        # print(glob.glob(str(path))) # DEBUG
    return input_entry


def get_prediction():
    # for scale_rate in scales:
    #     # print('scale rate: ', scale_rate)
    #     img = image.copy()
    #     img = scale(img, scale_rate)
    #     pred = np.zeros(img.shape[:2] + (num_class,))
    #     stride = img.shape[1]
    #     window_size = img.shape[:2]
    #     total = count_sliding_window(img, step=stride, window_size=wsize) // batch_size
    #     for i, coords in enumerate(
    #         tqdm(
    #             grouper(
    #                 batch_size,
    #                 sliding_window(img, step=stride, window_size=window_size),
    #             ),
    #             total=total,
    #             leave=False,
    #         )
    #     ):
    #
    #         image_patches = [
    #             np.copy(img[x : x + w, y : y + h]).transpose((2, 0, 1))
    #             for x, y, w, h in coords
    #         ]
    #         imgs_flip = [patch[:, ::-1, :] for patch in image_patches]
    #         imgs_mirror = [patch[:, :, ::-1] for patch in image_patches]
    #
    #         image_patches = np.concatenate(
    #             (image_patches, imgs_flip, imgs_mirror), axis=0
    #         )
    #         image_patches = np.asarray(image_patches)
    #         image_patches = torch.from_numpy(image_patches).cuda()
    pass


class Rx50InferenceTest(unittest.TestCase):
    def __init__(self):
        """
        https://github.com/qubvel/segmentation_models.pytorch/issues/371
        """
        super().__init__()
        checkpoint_path = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/MSCG-Rx50-epoch_6_loss_1.09903_acc_0.77739_acc-cls_0.53071_mean-iu_0.39789_fwavacc_0.64861_f1_0.53678_lr_0.0000845121.pth"
        # rx50 = get_net(checkpoint_path=checkpoint1)
        num_classes = 10
        node_size = (32, 32)
        model = load_rx50(num_classes=num_classes, node_size=node_size, model_path=checkpoint_path)

        # print(model.eval()) # DEBUG

        # model.load_state_dict(torch.load(PATH, )
        # test_files = get_real_test_list(bands=['NIR', 'RGB'])
        # test_files = load_test_img(test_files)
        # id_list = load_ids(test_files)
        # labels = land_classes
        # test_img = next(test_files)
        # print(type(test_img), test_img.size)

        # configure and prepare input image
        norm = True
        # load from file
        SUB_DIRS = ["boundaries", 'images/nir', 'images/rgb', 'masks']
        TEST_ROOT = Path(
            "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test/")
        ID = "ZTUK7EGVR_9431-3663-9943-4175"
        input_entry = get_input_entry(TEST_ROOT, ID)
        scale = 1.0
        im_rgb, im_nir, mask, bound = img_load(input_entry["rgb"], scale_rate=scale), \
                                      img_load(input_entry["nir"], scale_rate=scale), \
                                      img_load(input_entry["masks"], gray=True, scale_rate=scale), \
                                      img_load(input_entry["boundaries"], gray=True, scale_rate=scale)
        # perhaps inference requires a stack of images...
        # a collection -- a list because when thinking about a picture an a node...or maybe not...
        #
        print(json.dumps(input_entry, indent=4))
        with torch.no_grad():
            pred = model(torch.from_numpy(im_rgb))
        # img_path = (Path(
        #     "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test/images/rgb") / "ZTUK7EGVR_9431-3663-9943-4175.jpg")
        # # img_arr = np.im
        # img_p = Image.open(img_path).convert("RGB")
        # img_n = np.asarray(img_p, np.float32).transpose((2, 0, 1)) / 255.0
        #
        # # TODO: likely need to augment... b/c that's what the network learns from and is shift invariant of
        # #  at the core the net is learning
        # # img = np.asarray(pil_im, np.float32) / 255.0
        # # .transpose((2, 0, 1)) / 255.0
        # img_t = torch.from_numpy(img_n)
        # if norm:
        #     img_t = st.Normalize(*mean_std)(img_t)
        # print(type(img_t), img_t.shape)
        #
        # # img_t = torch.from_numpy(img)
        # batch_size = 1
        # shape_f = (batch_size, *img_t.shape)
        # # print(shape_f)
        # in_t = torch.reshape(img_t, shape_f)
        # print(in_t.shape)
        #
        # # img = np.asarray(pil_im, dtype='float32')
        # # apply transformations
        # # img = img_load(str(img_path), scale_rate=1)
        #
        # # img = img.cpu().numpy().transpose((1, 2, 0))
        # # print(img.size)
        # # img = torch.from_numpy(img)
        # # img = torch.reshape(img, (1,*img.shape))
        # # print(img.size())
        # start_time = time.time()
        # #
        # with torch.no_grad():
        #     pred = model(in_t)
        # end_time = time.time()
        # print("inference cost time: ", end_time - start_time)
        # #
        # # pred = np.argmax(pred, axis=-1)
        # output = pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
        # # for subdir in ['boundaries', 'masks']:
        # #     pred = pred * np.array(
        # #         cv2.imread(os.path.join(
        # #             '/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test', subdir,
        # #             id + '.png'), -1) / 255,
        # #         dtype=int)
        # #     print(os.path.join(
        # #         '/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test', subdir,
        # #         id + '.png'))
        # #
        # # filename = './{}.png'.format(id)
        # # output_path = "../submission"
        # # cv2.imwrite(os.path.join(output_path, filename), pred)
        #
        # # gtc = convert_from_color(gt)
        # # all_preds.append(pred)
        #
        # # all_gts.append(gt)
        # # ids.append(id)
        #
        # pass


class Rx101InferenceTest(unittest.TestCase):
    # checkpoint_path ="/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_4_loss_1.26896_acc_0.77713_acc-cls_0.54260_mean-iu_0.40996_fwavacc_0.64399_f1_0.55334_lr_0.0001245001.pth"
    # rx101 = get_net(checkpoint_path=checkpoint2)
    # rx101 = get_net(checkpoint_path=checkpoint2)
    pass


if __name__ == "__main__":
    unittest.main()
