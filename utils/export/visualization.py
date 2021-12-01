import glob
# from multiprocessing import Pool
import multiprocessing as mp
import os.path
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


class Visualize:
    @staticmethod
    def generate_entry(_input_paths, _prediction_paths) -> Tuple[int, ndarray, ndarray]:
        for idx, (in_path, pred_path) in enumerate(zip(_input_paths, _prediction_paths)):
            yield idx, np.asarray(Image.open(in_path)), np.asarray(Image.open(pred_path))
        pass

    # return in_pred_pil_gen

    @staticmethod
    def process_coloring(_input_data, _pred_data, _color_palette, _output_directory: Path, use_gpu):
        pred_colored = np.asarray(colorize_mask(_pred_data, COLOR_PALETTE).convert("RGB"))
        # op = _output_directory /
        pred_cv = pred_colored / 255.0

        # configure path and write image

        p = _output_directory / ""
        # pred_colored.save(_output_directory)
        cv2.imwrite(_output_directory / "")
        # add saving to the target output directory
        pass

    @staticmethod
    def multiprocess_visuals(input_paths: List[str], prediction_paths: List[str], processes: int,
                             output_directory: Path, verbose: bool = False):
        """Master process spawning worker processes for reading and writing of colorized visuals

        Test

        .. code-block: python

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

        :param processes:
        :param input_paths:
        :param prediction_paths:
        :return:
        """
        # data setup
        in_pred_pil_gen = Visualize.generate_entry(input_paths, prediction_paths)

        # multiprocess setup
        pool = mp.Pool(processes)
        processes = []

        # output directory and image writing setup
        basenames = [os.path.basename(p) for p in prediction_paths]
        subdirectories = [
            os.path.splitext(p[0])[0] for p in basenames
        ]
        for subdir in subdirectories:
            p = (Path(output_directory) / subdir)
            if not os.path.exists(p):
                os.mkdir(p)
                if verbose: print("Created:", p)

        # start processes
        for idx, input_data, pred_data in tqdm.tqdm(in_pred_pil_gen):
            pass
        #     out_dir = None
        #     p = mp.Process(target=Visualize.process_coloring, args=(
        #         input_data,
        #         pred_data,
        #         COLOR_PALETTE,
        #         # out_path,
        #         # False
        #         # True
        #     ))
        #     processes.append(p)
        #     p.start()
        #     pass
        #
        # # collect processes
        # for p in processes:
        #     p.join()

        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pred_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
    # in_paths = glob.glob(str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / f"{}.jpg".format(basename)))
    in_paths = [
        str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / "{}.jpg".format(
            os.path.splitext(
                os.path.basename(fp))[0])
            ) for fp in pred_paths
    ]
    # get the generator
    _ = Visualize.multiprocess_visuals(input_paths=in_paths, prediction_paths=pred_paths, processes=os.cpu_count(),
                                       verbose=True)
    # idx, in_arr, pred_arr = next(_)
    # # apply the coloring
    # pred_colored = np.asarray(colorize_mask(pred_arr, COLOR_PALETTE).convert("RGB"))
    #
    # # sanity check with visualization
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(in_arr)
    # ax[1].imshow(pred_colored)
    # plt.show()
