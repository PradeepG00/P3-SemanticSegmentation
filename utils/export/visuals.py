import glob
from pathlib import Path
from typing import List

import numpy as np
import tqdm
from PIL import Image
import torchvision.transforms as standard_transforms

# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


def multiprocess_visuals(input_paths: List[str], prediction_paths: List[str]):
    """Master process spawning worker processes for reading and writing of colorized visuals

    :param input_paths:
    :param prediction_paths:
    :return:
    """
    from multiprocessing import Pool

    def generate_entry(_input_paths, _prediction_paths):
        for idx, (in_path, pred_path) in enumerate(zip(_input_paths, _prediction_paths)):
            yield idx, np.asarray(Image.open(in_path)), np.asarray(Image.open(pred_path))
        pass

    in_pred_pil_gen = generate_entry(input_paths, prediction_paths)
    return in_pred_pil_gen

    def process_coloring(_input_data, _pred_data, _color_palette, _output_directory: str, use_gpu):
        pass

    processes = []
    in_pred_pil_gen = generate_entry(input_paths, prediction_paths)
    for idx, input_data, pred_data in tqdm.tqdm(in_pred_pil_gen):
        input_pil = colorize_mask(input_data, COLOR_PALETTE)
        pred_pil = colorize_mask(pred_data, COLOR_PALETTE)

        pass

    pass


if __name__ == "__main__":
    input_paths = glob.glob(str(Path(DATASET_ROOT) / "test" / "images" / "rgb" / "*.jpg"))
    print(len(input_paths))
    prediction_paths = glob.glob(str(Path(PROJECT_ROOT) / "submission" / "results" / "*.png"))
    print(len(prediction_paths))
    _ = multiprocess_visuals(input_paths=input_paths, prediction_paths=prediction_paths)
    idx, in_arr, pred_arr = next(_)
    print(pred_arr.shape)
    # print(idx, np.asarray(in_arr), np.asarray(pred_arr))
    pred_colored = np.asarray(colorize_mask(pred_arr, COLOR_PALETTE).convert("RGB"))
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(pred_colored)
    ax[1].imshow(pred_colored)
    plt.show()
    # print(colored.convert("RGB"))
    # print(type(colored))

    print(pred_colored.shape)
    # print(colored[])
