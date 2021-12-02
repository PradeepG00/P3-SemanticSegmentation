from __future__ import division

import itertools
import os
import time
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torchvision.transforms as st
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from core import INFERENCE_PATH
from utils import PROJECT_ROOT, DEBUG, check_mkdir
from utils.data import LAND_CLASSES
from utils.data import TEST_DIR, IDS
from utils.data.augmentation import scale
from utils.data.dataset import get_real_test_list
from utils.data.preprocess import MEAN_STD
# from utils.data import mean_std
from utils.trace.checkpoint import (
    get_net,
    checkpoint1,
    checkpoint2,
    load_test_img,
    load_ids,
)


# USE_GPU = True
# model_name = "ensemble"


def main():
    check_mkdir(INFERENCE_PATH)
    nets = []
    cpath_2 = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_20_loss_1.26717_acc_0.78944_acc-cls_0.55146_mean-iu_0.40546_fwavacc_0.66615_f1_0.54439_lr_0.0000946918.pth"
    cpath_1 = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/MSCG-Rx50-epoch_33_loss_1.08814_acc_0.79293_acc-cls_0.59024_mean-iu_0.41186_fwavacc_0.67505_f1_0.55180_lr_0.0000372098.pth"
    net1 = get_net(checkpoint=checkpoint1,
                   use_gpu=False,
                   # use_gpu=USE_GPU,
                   checkpoint_path=cpath_1)  # MSCG-Net-R50
    # print(net1.eval())
    net2 = get_net(
        checkpoint=checkpoint2,
        use_gpu=False,
        # use_gpu=USE_GPU,
        checkpoint_path=cpath_2
    )  # MSCG-Net-R101
    net2.name = "MSCG-Net-R101"
    net1.name = "MSCG-Net-R50"
    net2.eval()
    net1.eval()
    # net3 = get_net(checkpoint3)  # MSCG-Net-R101
    # enable GPU
    # net1.cuda(0)
    # net2.cuda(0)
    # print(net2.eval())

    nets.append(net1)
    nets.append(net2)
    # nets.append(net3)

    # checkpoint1 + checkpoint2, test score 0.599,
    # checkpoint1 + checkpoint2 + checkpoint3, test score 0.608

    test_files = get_real_test_list(bands=["NIR", "RGB"])
    run_tta_real_test(
        nets,
        stride=600,
        batch_size=1,
        norm=False,
        window_size=(512, 512),
        labels=LAND_CLASSES,
        test_set=test_files,
        all=True,
    )


def run_tta_real_test(
        nets,
        all=False,
        labels=LAND_CLASSES,
        norm=False,
        test_set=None,
        stride=600,
        batch_size=5,
        window_size=(512, 512),
        # use_gpu: bool=False
):
    """Run test time augmentation using the test set and the model ensemble

    :param nets:
    :param all:
    :param labels:
    :param norm:
    :param test_set:
    :param stride:
    :param batch_size:
    :param window_size:
    # :param use_gpu:
    :return:
    """
    time_0 = time.time()
    print(labels, norm, stride, batch_size, window_size)
    # Setup data generators
    gen_test_image = load_test_img(test_set)
    gen_ids = load_ids(test_set)
    predictions = []
    num_class = len(labels)
    ids = []
    total_ids = 0

    for k in test_set[IDS].keys():
        total_ids += len(test_set[IDS][k])
    print("total ids:", total_ids)

    times_dict = {
        # id : {
        # id: {
        #     "preprocess": 0,
        #     "inference": 0,
        #     "boundaries": 0,
        #     "masks": 0,
        #     "full_iteration": 0
        # }
    }

    # step through each sample
    for img, id in tqdm(zip(gen_test_image, gen_ids), total=total_ids, leave=False):
        times_dict[id] = {}
        if DEBUG:
            print("Start preprocessing...")
            print(type(img))
            print(img.shape)
        start_time = time.time()  # TIMER
        # read input as numpy H,W,C
        img = np.asarray(img, dtype="float32")

        if DEBUG:
            print(type(img))
            print("--- Start Input NumPy Shape:", img.shape)

        #  convert to tensor
        img = st.ToTensor()(img)
        if DEBUG:
            print("Applied convert to", type(img))
            print("Input Tensor Shape:", img.shape)
            print("Input Tensor Shape: {}".format(img.shape))  # DEBUG

        # apply normalization in tensor format
        img = img / 255.0
        if norm:
            img = st.Normalize(*MEAN_STD)(img)
        img = img.cpu().numpy().transpose(
            (1, 2, 0))  # after applying the necessary transformations we go to 3-dims numpy 512,512,4... why?

        if DEBUG:
            print("-- End Post-Transpose as NumPy Size:", img.shape)
            end_time = time.time()  # TIMER
            time_delta = end_time - start_time
            print("Preprocessing time:", time_delta)

        # times_dict[id]["preprocess"] = time_delta
        # start_time = time.time()

        # Inference
        with torch.no_grad():
            print("Running fusion prediction")
            pred = fusion_prediction(
                nets,
                image=img,
                scales=[
                    # 1.2,
                    1.0],
                # TODO: what happens when we do variations? >> it breaks down b/c of the broadcassting of shapes
                batch_size=batch_size,
                num_class=num_class,
                window_size=window_size,
            )
        # end_time = time.time()
        # time_delta = end_timedraw - start_time
        if DEBUG:
            print("inference cost time: ", end_time - start_time)
            print("inference cost time: {}".format(end_time - start_time))

        # times_dict[id]["inference"] = time_delta
        # start_time = time.time()

        pred = np.argmax(pred, axis=-1)
        # if DEBUG:
        print("Prediction shape:", pred.shape)

        # bounding box? image segmentation / mask ie a filter that defines
        for key in ["boundaries", "masks"]:
            pred = pred * np.array(
                cv2.imread(
                    os.path.join(
                        (PROJECT_ROOT / TEST_DIR), key, id + ".png", ), -1, ) / 255, dtype=int, )
            if key == "boundaries":
                end_time = time.time()
                time_delta = end_time - start_time
                # times_dict[id]["masks"]
                # start_time = time.time()
            else:
                pass
                # end_time = time.time()
                # time_delta = end_time - start_time
                # times_dict[id]["boundaries"]
        filename = "./{}.png".format(id)
        print(f"writing prediction to: {os.path.join(INFERENCE_PATH, filename)}")
        cv2.imwrite(os.path.join(INFERENCE_PATH, filename), pred)

        # gtc = convert_from_color(gt)
        predictions.append(pred)

        # all_gts.append(gt)
        ids.append(id)
        time_f = time.time()
        times_dict[id]["full_iteration"] = (time_f - time_0)
    # accuracy, cm = metrics(np.concatenate([p.ravel() for p in all_preds]),
    #                        np.concatenate([p.ravel() for p in all_gts]).ravel(), label_values=labels)
    with open("inference_timings.json", "w") as fh:
        import json
        json.dump(times_dict, fh, indent=4)
    if all:
        return predictions, ids
    else:
        return predictions


def metrics(predictions, gts, label_values=LAND_CLASSES):
    """

    :param predictions:
    :param gts:
    :param label_values:
    :return:
    """
    cm = confusion_matrix(gts, predictions, range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    # Compute F1 score
    f1_score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            f1_score[i] = 2.0 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except BaseException:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(f1_score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    return accuracy, cm


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    # print(top.shape[0])
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    print("total number of sliding windows: ", c)
    return c


def grouper(n, iterable):
    """

    :param n:
    :param iterable:
    :return:
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def fusion_prediction(nets, image, scales, batch_size=1, num_class=7, window_size=(512, 512), use_gpu: bool = False):
    """Ensemble method combining the outputs of the SCG-GCN module Rx50 and Rx101 for image segmentation inference

    .. math::

        (\\hat{y}+Z^{(2)})

    1. Extraction of patches defining the pertaining to the sliding window
    2. Reversal of rotations
    3. Projection to 2D-space

    :param use_gpu:
    :param nets:
    :param image:
    :param scales:
    :param batch_size:
    :param num_class:
    :param window_size:
    :return:
    """
    # initialize an empty tensor 512x512x10 or H x W x Classes
    pred_all = np.zeros(image.shape[:2] + (num_class,))
    print(f"shape of all predictions: {pred_all.shape}, number of classes: {num_class}")

    # for each rate of scale
    for scale_rate in scales:
        patch_dict = get_kernel_patches(image, scale_rate, num_class, window_size, batch_size)
        img, strides, window_size, total, pred = patch_dict["img"], patch_dict["strides"], patch_dict["win_size"], \
                                                 patch_dict["total"], patch_dict["pred"]
        # print("prediction pred.shape
        print(pred.shape)  #

        for i, coords in enumerate(
                # tqdm(
                grouper(
                    batch_size,
                    sliding_window(
                        img,
                        step=strides,
                        window_size=window_size),
                ),
                #     total=total,
                #     leave=False,
                # )
        ):
            start_time = time.time()

            print(i, coords)
            # not sure if this is... ?
            # looks like the original image... or the copied image
            # thus how many times do we do the grouping...
            # TODO: what is the role of grouper

            # handle for the 2 other variations to form the
            # 3-views
            image_patches = [
                np.copy(img[x: x + w, y: y + h]).transpose((2, 0, 1))
                for x, y, w, h in coords
            ]
            imgs_flip = [patch[:, ::-1, :] for patch in image_patches]
            imgs_mirror = [patch[:, :, ::-1] for patch in image_patches]

            image_patches = np.concatenate(
                (image_patches, imgs_flip, imgs_mirror), axis=0
            )
            image_patches = np.asarray(image_patches)
            if use_gpu:
                image_patches = torch.from_numpy(image_patches).cuda()
            else:
                image_patches = torch.from_numpy(image_patches).cpu()

            # net output fusing here
            for net in nets:
                outs = net(
                    image_patches
                )  # + Fn.torch_rot270(core(Fn.torch_rot90(image_patches)))
                # outs = core(image_patches)
                outs = outs.data.cpu().numpy()
                print(f"Net {net.name} Output shape: {outs.shape}")  # (3, 10, 512, 512)
                b, _, _, _ = outs.shape

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs[0: b // 3, :, :, :], coords):
                    print("Pre-transpose:",
                          out.shape)  # 3 batches for the 3 tranforms of the input, 10 for classes 512x512 b/c input HxW
                    out = out.transpose((1, 2, 0))
                    print("Post-transpose:", out.shape)
                    pred[x: x + w, y: y + h] += out
                    print("Prediction")
                    print(pred[x: x + w, y: y + h])

                for out, (x, y, w, h) in zip(
                        outs[
                        b // 3: 2 * b // 3,
                        :, :, :
                        ], coords
                ):
                    out = out[:, ::-1, :]  # flip back
                    out = out.transpose((1, 2, 0))
                    pred[x: x + w, y: y + h] += out

                for out, (x, y, w, h) in zip(outs[2 * b // 3: b, :, :, :], coords):
                    out = out[:, :, ::-1]  # mirror back
                    out = out.transpose((1, 2, 0))
                    pred[x: x + w, y: y + h] += out

                del outs
            end_time = time.time()

        pred_all += scale(pred, 1.0 / scale_rate)

    return pred_all


def get_kernel_patches(image, scale_rate, num_class, wsize, batch_size) -> dict:
    """

    :param image:
    :param scale_rate:
    :param num_class:
    :param wsize:
    :param batch_size:
    :return:
    """
    img = image.copy()
    print(f"shape of input: {img.shape}")
    img = scale(img, scale_rate)
    print(f"post-scaling -- {scale_rate} | shape of input: {img.shape}")
    pred = np.zeros(img.shape[:2] + (num_class,))
    print(f"shape of prediction: {pred.shape}")

    stride = img.shape[1]
    print(f"stride size: {stride}")

    window_size = img.shape[:2]
    print(f"kernel size: {window_size}")
    # count the total sliding windows -- invert...?  from cnn
    total = count_sliding_window(img, step=stride,
                                 window_size=wsize) // batch_size  # TODO: why do we count the sliding windows?
    # thus we have the output signal ...?
    print("fusing outputs and up-sampling")
    rv = {
        "img": img,
        # img, stride, window_size, total, pred
        "strides": stride,
        "win_size": window_size,
        "total": total,
        "pred": pred
    }
    return rv


def fuse_and_up_sample():
    pass


def inference(img: np.array, boundaries, masks,
              labels: List[str], num_class: int, window_size: Tuple[int],
              model_paths: dict = dict(
                  rx50="/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/MSCG-Rx50-epoch_33_loss_1.08814_acc_0.79293_acc-cls_0.59024_mean-iu_0.41186_fwavacc_0.67505_f1_0.55180_lr_0.0000372098.pth",
                  rx101="/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_20_loss_1.26717_acc_0.78944_acc-cls_0.55146_mean-iu_0.40546_fwavacc_0.66615_f1_0.54439_lr_0.0000946918.pth"),
              batch_size: int = 1,
              normalize: bool = True,
              models_selected: str = "all",
              # use_gpu: bool = False,
              all: bool = True
              ):
    """

    :param img: input NumPy array image of shape (512,512,4)
    :param model_paths:
    :param num_class:
    :param window_size:
    :param batch_size:
    :param models_selected:
    :param use_gpu:
    :return:
    """
    assert models_selected.lower() in ["all", "rx50", "rx101", "both"]
    # print(labels, norm, stride, batch_size, window_size)
    # Setup data generators
    # gen_test_image = load_test_img(test_set)
    # gen_ids = load_ids(test_set)
    predictions = []
    num_class = len(labels)
    ids = []
    total_ids = 0
    #
    # cpath_2 =
    # cpath_1 =

    # load models from model_paths
    nets = []
    rx50_net = get_net(checkpoint=checkpoint1,
                       use_gpu=False,
                       # use_gpu=use_gpu,
                       checkpoint_path=model_paths["rx50"])  # MSCG-Net-R50
    print(rx50_net.eval())
    rx101_net = get_net(checkpoint=checkpoint2,
                        use_gpu=False,
                        # use_gpu=use_gpu,
                        checkpoint_path=model_paths["rx101"])  # MSCG-Net-R101
    # net3 = get_net(checkpoint3)  # MSCG-Net-R101
    print(rx101_net.eval())  # DEBUG
    if models_selected.lower() == "all":
        nets.append(rx50_net)
        nets.append(rx101_net)
    elif models_selected.lower() == "rx50":
        nets.append(rx50_net)
    elif models_selected.lower() == "rx101":
        nets.append(rx101_net)
    else:
        raise ValueError("Model selected does not exist")

    # img = preprocess_for_inference(img)
    # Preprocess: standardize, normalize, transpose
    # org_img = img.copy()  # for visualization
    print("Start preprocessing...")
    print(type(img))
    print(img.shape)
    img = np.asarray(img, dtype="float32")
    print(type(img))
    print("--- Start Input NumPy Shape:", img.shape)
    print("Applied convert to", type(img))
    print("Input Tensor Shape:", img.shape)
    # print(
    #     "Input Tensor Shape: {}".format(img.shape))  # DEBUG
    img = img / 255.0
    # if DEBUG:
    #     print("img / 255 = ", img);
    # print("img / 255 = ", img)  # DEBUG# DEBUG

    # print(f"normalizing image: {img}")
    img = st.ToTensor()(img)
    if normalize:
        img = st.Normalize(*MEAN_STD)(img)
        normalized = True
    img = img.cpu().numpy().transpose(
        (2, 1, 0))  # after applying the necessary transformations we go to 3-dims numpy 512,512,4... why?
    if DEBUG: print("-- End Post-Transpose as NumPy Size:", img.shape)

    with torch.no_grad():
        print("Running fusion prediction")
        start_time = time.time()
        pred = fusion_prediction(
            nets,
            image=img,
            scales=[
                # 1.2,
                1.0],
            # TODO: what happens when we do variations? >> it breaks down b/c of the broadcassting of shapes
            batch_size=batch_size,
            num_class=num_class,
            window_size=window_size,
        )
    end_time = time.time()
    print("inference cost time: {}".format(end_time - start_time))

    pred = np.argmax(pred, axis=-1)

    # bounding box? image segmentation / mask ie a filter that defines
    for sub_dir in ["boundaries", "masks"]:
        path = os.path.join(
            (PROJECT_ROOT / TEST_DIR),
            sub_dir,
            id + ".png",
        )
        print(path)
        pred = pred * np.array(
            cv2.imread(
                os.path.join(
                    (PROJECT_ROOT / TEST_DIR),
                    sub_dir,
                    id + ".png",
                ), -1, ) / 255,
            dtype=int,
        )
    filename = "./{}.png".format(id)
    print(f"writing prediction to: {os.path.join(INFERENCE_PATH, filename)}")
    cv2.imwrite(os.path.join(INFERENCE_PATH, filename), pred)

    # gtc = convert_from_color(gt)
    predictions.append(pred)

    # all_gts.append(gt)
    ids.append(id)

    # accuracy, cm = metrics(np.concatenate([p.ravel() for p in all_preds]),
    #                        np.concatenate([p.ravel() for p in all_gts]).ravel(), label_values=labels)
    if all:
        return predictions, ids
    else:
        return predictions
    pass


def preprocess_for_inference(img: np.array, norm: bool = True,
                             # use_gpu: bool=False
                             ):
    # Preprocess: standardize, normalize, transpose
    # org_img = img.copy()  # for visualization
    print("Start preprocessing...")
    print(type(img))
    print(img.shape)
    img = np.asarray(img, dtype="float32")
    print(type(img))
    print("--- Start Input NumPy Shape:", img.shape)
    img = st.ToTensor()(img)
    print("Applied convert to", type(img))
    print("Input Tensor Shape:", img.shape)
    print(
        "Input Tensor Shape: {}".format(img.shape))  # DEBUG
    img = img / 255.0
    # if DEBUG:
    #     print("img / 255 = ", img);
    # print("img / 255 = ", img)  # DEBUG# DEBUG
    if norm:
        img = st.Normalize(*MEAN_STD)(img)
        # print(f"normalizing image: {img}")
    img = img.cpu().numpy().transpose(
        (1, 2, 0))  # after applying the necessary transformations we go to 3-dims numpy 512,512,4... why?
    if DEBUG: print("-- End Post-Transpose as NumPy Size:", img.shape)
    return img


if __name__ == "__main__":
    main()
