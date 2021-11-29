import os
import time
from pathlib import Path
import cv2
import torch
import unittest
import numpy as np
from PIL import Image

import train_R50
from models.mscg import get_model
from utils.data.augmentations import img_load
from utils.data.preprocess import get_real_test_list, land_classes
from utils.trace.checkpoint import get_net, checkpoint2, checkpoint1, load_test_img, load_ids
import torchvision.transforms as st
from utils.data.visual import mean_std


class Rx50InferenceTest(unittest.TestCase):
    checkpoint_path = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/MSCG-Rx50-epoch_6_loss_1.09903_acc_0.77739_acc-cls_0.53071_mean-iu_0.39789_fwavacc_0.64861_f1_0.53678_lr_0.0000845121.pth"
    # rx50 = get_net(checkpoint_path=checkpoint1)
    _model = get_model(node_size=(32, 32), classes=10)
    model, _ = train_R50.train_args.resume_train(
        _model,
        checkpoint_path=checkpoint_path,
        use_gpu=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)
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
    img_path = Path(
        "/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test/images/rgb") / "ZTUK7EGVR_9431-3663-9943-4175.jpg"
    # img_arr = np.im
    pil_im = Image.open(img_path).convert("RGB")
    img = np.asarray(pil_im, np.float32).transpose((2, 0, 1)) / 255.0
    img = st.ToTensor()(img)
    if norm:
        img = st.Normalize(*mean_std)(img)
    img_t = torch.from_numpy(img)
    batch_size = 1
    shape_f = (batch_size, *img_t.shape)
    print(shape_f)
    img_t = torch.reshape(img_t, (batch_size, img_t.shape))
    print(img_t.shape)
    #
    # img = np.asarray(pil_im, dtype='float32')
    # apply transformations
    # img = img_load(str(img_path), scale_rate=1)

    # img = img.cpu().numpy().transpose((1, 2, 0))
    # print(img.size)
    # img = torch.from_numpy(img)
    # img = torch.reshape(img, (1,*img.shape))
    # print(img.size())
    start_time = time.time()

    with torch.no_grad():
        pred = _model(img)
    end_time = time.time()
    print('inference cost time: ', end_time - start_time)

    pred = np.argmax(pred, axis=-1)

    # for subdir in ['boundaries', 'masks']:
    #     pred = pred * np.array(
    #         cv2.imread(os.path.join(
    #             '/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test', subdir,
    #             id + '.png'), -1) / 255,
    #         dtype=int)
    #     print(os.path.join(
    #         '/home/hanz/github/agriculture-vision-datasets/2021/supervised/Agriculture-Vision-2021/test', subdir,
    #         id + '.png'))
    #
    # filename = './{}.png'.format(id)
    # output_path = "../submission"
    # cv2.imwrite(os.path.join(output_path, filename), pred)

    # gtc = convert_from_color(gt)
    # all_preds.append(pred)

    # all_gts.append(gt)
    # ids.append(id)

    pass


class Rx101InferenceTest(unittest.TestCase):
    # checkpoint_path ="/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_4_loss_1.26896_acc_0.77713_acc-cls_0.54260_mean-iu_0.40996_fwavacc_0.64399_f1_0.55334_lr_0.0001245001.pth"
    # rx101 = get_net(checkpoint_path=checkpoint2)
    # rx101 = get_net(checkpoint_path=checkpoint2)
    pass


if __name__ == "__main__":
    unittest.main()
