import glob
# from multiprocessing import Pool
import multiprocessing
import multiprocessing as mp
import os.path
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image
import torchvision.transforms as standard_transforms
from numpy import ndarray

from utils import PROJECT_ROOT
from utils.data import COLOR_PALETTE, DATASET_ROOT
from utils.data.dataset import get_real_test_list


# @staticmethod
def multiprocess_visuals(
        id_list: List[str],
        input_paths: List[str], prediction_paths: List[str], processes: int,
        output_directory: Path, verbose: bool = False
):
    """Master process spawning worker processes for reading and writing of colorized visuals

    Test

    .. code-block: python

        test_path_dict = get_real_test_list(bands=["NIR", "RGB"])  # ../test/images/rgb/{}.jpg
        pred_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
        out_dir = str(Path(PROJECT_ROOT) / "submission/visualized/")
        im_rgb_path_fmt = str(test_path_dict["images"][1])

        # aggregate prediction ids
        pred_ids = [
            "{}".format(
                os.path.splitext(
                    os.path.basename(fp))[0])
            for fp in pred_paths
        ]

        input_paths = []
        for id in pred_ids:
            p = im_rgb_path_fmt.format(id)
            input_paths.append(p)
            if not os.path.exists(Path(out_dir) / id):
                print("Created: ", Path(out_dir) / id)
                os.mkdir(Path(out_dir) / id)

        multiprocess_visuals(
            id_list=pred_ids,
            input_paths=input_paths,
            prediction_paths=pred_paths,
            output_directory=out_dir,
            verbose=True,
            processes=os.cpu_count()
        )

        # plot and visualize
        import matplotlib.pyplot as plt
        input_paths = glob.glob(str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / "*.jpg"))
        prediction_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
        _ = multiprocess_visuals(input_paths=input_paths, prediction_paths=prediction_paths)

        idx, in_arr, pred_arr = next(_)
        pred_colored = np.asarray(colorize_mask(pred_arr, COLOR_PALETTE).convert("RGB"))
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(in_arr)
        ax[1].imshow(pred_colored)
        plt.show()

    :param output_directory:
    :param processes:
    :param input_paths:
    :param prediction_paths:
    :return:
    """
    # data setup
    in_pred_pil_gen = generate_entry(input_paths, prediction_paths, verbose=True)

    idx, input_data, pred_data = next(in_pred_pil_gen)
    id = id_list[idx]
    print(id_list[idx], input_paths[idx], prediction_paths[idx])
    assert (id in input_paths[idx]) and (id in prediction_paths[idx])

    # multiprocess setup
    # pool = mp.Pool(processes)
    # sema = mp.Semaphore(processes)
    procs = []
    import psutil
    import subprocess as sp

    # create and start processes
    for idx, input_data, pred_data in tqdm.tqdm(in_pred_pil_gen):
        # mapping of output paths
        out_paths_dict = {
            "in_rgb": str(Path(output_directory) / id / ("input.jpg")),
            "pred_src": str(Path(output_directory) / id / ("lut.png")),
            "pred_rgb": str(Path(output_directory) / id / ("lut_rgb.png"))
        }
        id = id_list[idx]
        if verbose: print(id_list[idx], input_paths[idx], prediction_paths[idx])
        assert (id in input_paths[idx]) and (id in prediction_paths[idx])

        p = mp.Process(target=apply_color_and_save, args=(
            # sema,
            None,
            input_data,
            pred_data,
            COLOR_PALETTE,
            out_paths_dict,
            False,
            True
        ))
        # print()
        procs.append(p)
        # if idx == 0:
        #     apply_color_and_save(
        #         sema=None,
        #         _input_data=input_data,
        #         _pred_data=pred_data,
        #         _color_palette=COLOR_PALETTE,
        #         _output_paths=out_paths_dict,
        #         use_gpu=False,
        #         verbose=True
        #     )
        #     input()
        p.start()

    # for p in procs:
        # c += 1
        # if c == 5:
        #     break
        # pass
    for p in procs:
        p.join()


class DeNormalize(object):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_visualize(args):
    visualize = standard_transforms.Compose(
        [
            standard_transforms.Resize(300),
            standard_transforms.CenterCrop(300),
            standard_transforms.ToTensor(),
        ]
    )

    if args.pre_norm:
        restore = standard_transforms.Compose(
            [DeNormalize(*DeNormalize.mean_std), standard_transforms.ToPILImage(), ]
        )
    else:
        restore = standard_transforms.Compose([standard_transforms.ToPILImage(), ])

    return visualize, restore


def setup_palette(palette):
    """

    :param palette:
    :return:
    """
    palette_rgb = []
    for _, color in palette.items():
        palette_rgb += color

    zero_pad = 256 * 3 - len(palette_rgb)

    for i in range(zero_pad):
        palette_rgb.append(0)
    return palette_rgb


def colorize_mask(mask, palette):
    """Color code for the mask

    :param mask:
    :param palette:
    :return:
    """
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(setup_palette(palette))
    return new_mask


def convert_to_color(arr_2d, palette):
    """

    :param arr_2d:
    :param palette:
    :return:
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d


# class Visualize:
# @staticmethod
def generate_entry(_input_paths, _prediction_paths, verbose: bool = False) -> Tuple[int, ndarray, ndarray]:
    for idx, (in_path, pred_path) in enumerate(zip(_input_paths, _prediction_paths)):
        if verbose: print("Yield from:", in_path, pred_path)
        yield idx, np.asarray(Image.open(in_path)), np.asarray(Image.open(pred_path))
    pass


# return in_pred_pil_gen

# @staticmethod
def apply_color_and_save(
        sema: multiprocessing.Semaphore,
        _input_data, _pred_data,
        _color_palette, _output_paths: dict[Path],
        use_gpu: bool, verbose: bool = True,
) -> None:
    """Function for applying color to a provided mask array and writing to disk

    :param sema:
    :param verbose:
    :param _input_data:
    :param _pred_data:
    :param _color_palette:
    :param _output_paths: desired output paths
    :param use_gpu:
    :return:
    """
    if sema: sema.acquire()
    pred_colored = np.asarray(
        colorize_mask(_pred_data, COLOR_PALETTE).convert("RGB")
    )
    # op = _output_directory /
    # pred_colored = pred_colored / 255.0

    # configure path and write image
    pred_src_path = str(_output_paths["pred_src"])
    pred_color_path = str(_output_paths["pred_rgb"])
    in_path = str(_output_paths["in_rgb"])
    if verbose:
        print("Writing to...")
        print(pred_src_path, pred_color_path, in_path)
    # pred_colored.save(_output_directory)
    im = Image.fromarray(pred_colored)
    im.save(pred_color_path)
    im = Image.fromarray(_pred_data)
    im.save(pred_src_path)
    im = Image.fromarray(_input_data)
    im.save(in_path)

    # im.sa
    # cv2.imwrite(pred_color_path, pred_colored)
    # cv2.imwrite(pred_src_path, _pred_data)
    # cv2.imwrite(in_path, _input_data)
    if sema:
        sema.release()
        # time.sleep(3)
    # add saving to the target output directory
    return pred_colored


#     out_dir = None
#
# # collect processes
# for p in processes:
#     p.join()
def run_visualization_demo(

) -> None:
    test_path_dict = get_real_test_list(bands=["NIR", "RGB"])  # ../test/images/rgb/{}.jpg
    pred_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
    out_dir = str(Path(PROJECT_ROOT) / "submission/visualized/")
    im_rgb_path_fmt = str(test_path_dict["images"][1])

    # aggregate prediction ids
    pred_ids = [
        "{}".format(
            os.path.splitext(
                os.path.basename(fp))[0])
        for fp in pred_paths
    ]
    print(pred_ids[:5])
    print(im_rgb_path_fmt)
    # in_paths = glob.glob(str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / f"{}.jpg".format(basename)))
    print(os.path.exists(out_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    input_paths = []
    for id in pred_ids:
        p = im_rgb_path_fmt.format(id)
        input_paths.append(p)
        if not os.path.exists(Path(out_dir) / id):
            print("Created: ", Path(out_dir) / id)
            os.mkdir(Path(out_dir) / id)

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    # input_paths = glob.glob(str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / "*.jpg"))
    prediction_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
    im_pred_pil_gen = generate_entry(input_paths, pred_paths, verbose=True)

    # _ = multiprocess_visuals(
    #     input_paths=input_paths, prediction_paths=prediction_paths
    # )
    ROWS = 4
    f, ax = plt.subplots(ROWS, 2)
    # plt.rcParams["figure.figsize"] = (20, 3)
    i = 0
    for idx, in_arr, pred_arr in im_pred_pil_gen:
        print(i)
        if (i + 1) == ROWS + 1:
            plt.tight_layout()
            plt.show()
            # reset
            f, ax = plt.subplots(ROWS, 2)
            i = 0
            input()
        idx, in_arr, pred_arr = next(im_pred_pil_gen)
        pred_colored = np.asarray(colorize_mask(pred_arr, COLOR_PALETTE).convert("RGB"))
        ax[i, 0].imshow(in_arr, interpolation='nearest')
        ax[i, 1].imshow(pred_colored, interpolation='nearest')
        i += 1


if __name__ == "__main__":
    pass
    #
    test_path_dict = get_real_test_list(bands=["NIR", "RGB"])  # ../test/images/rgb/{}.jpg
    pred_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
    out_dir = str(Path(PROJECT_ROOT) / "submission/visualized/")
    im_rgb_path_fmt = str(test_path_dict["images"][1])

    # aggregate prediction ids
    pred_ids = [
        "{}".format(
            os.path.splitext(
                os.path.basename(fp))[0])
        for fp in pred_paths
    ]
    print(pred_ids[:5])
    print(im_rgb_path_fmt)
    # in_paths = glob.glob(str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / f"{}.jpg".format(basename)))
    print(os.path.exists(out_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    input_paths = []
    for id in pred_ids:
        p = im_rgb_path_fmt.format(id)
        input_paths.append(p)
        od = Path(out_dir) / id
        if not os.path.exists(Path(out_dir) / id):
            print("Created: ", Path(out_dir) / id)
            os.mkdir(Path(out_dir) / id)
        else:
            print("Existing directory:", od)
            print(os.listdir(od))
        print("===================")

    # print(os.path.exists(input_paths[0]))
    # print(len(input_paths) == len(pred_paths) == len(pred_ids))

    # single process and write
    # apply_color_and_save(
    #     sema=None,
    #     input_data,
    #     pred_data,
    #     COLOR_PALETTE,
    #     out_paths_dict,
    #     False,
    #     True
    # )

    #
    # sys.exit(0)
    p = os.cpu_count()
        # - 4
    # p = 7
    print("Max processes:", p)
    multiprocess_visuals(
        id_list=pred_ids,
        input_paths=input_paths,
        prediction_paths=pred_paths,
        output_directory=out_dir,
        verbose=True,
        processes=p
    )
