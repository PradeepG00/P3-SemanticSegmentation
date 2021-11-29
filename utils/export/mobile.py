from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.mscg import RX101GCN3Head4Channel
from models.mscg import RX50GCN3Head4Channel


def convert_to_mobile(model: str, source_path: str or Path, output_path: str or Path, num_classes: int) -> nn.Module:
    """Main function for converting MSCG core to PyTorch Mobile

    **NOTE** Usage of PyTorch Mobile to convert the MSCG-Nets requires usage of a matching Android PyTorch Mobile Version 1.10

    :param num_classes:
    :param model:
    :param source_path:
    :param output_path:
    :return:
    """
    if model.lower() == "rx50":
        return _convert_rx50_to_mobile(source_path, output_path, num_classes)
    elif model.lower() == "rx100":
        return _convert_rx101_to_mobile(source_path, output_path, num_classes)


def _convert_rx50_to_mobile(model_path: str, num_classes: int,
                            output_path: str or Path = "./rx50_optimized_scripted1.ptl"):
    """Function handling conversion of MSCG-Net Rx50 Models

    Models are loaded using the CPU for conversion to PyTorch Mobile

    :param model_path: path defining the source of the model's .pt pytorch file
    :param output_path: target path to write .ptl
    :param num_classes:
    :return:
    """

    model = RX50GCN3Head4Channel(out_channels=num_classes)
    for param in model.parameters():
        param.requires_grad = False

    # print(model.graph_layers2.fc[0])
    # Add on classifier
    model.graph_layers2.fc[0] = nn.Linear(128, num_classes)
    # print(metrics)
    model.scg.mu[0] = nn.Conv2d(1024, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.scg.logvar[0] = nn.Conv2d(1024, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    # print(model.eval())
    # summary(metrics,input_size=(4,512,512))
    # print(summary(metrics,(10,4,32,32)))
    example = torch.rand(10, 4, 32, 32)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("./rx50_jit_traced.pt")

    # normal .ptl file
    # traced_script_module._save_for_lite_interpreter("./rx50_scripted1.ptl")

    optimized_scripted_module = optimize_for_mobile(traced_script_module)
    # optimized .ptl file
    optimized_scripted_module._save_for_lite_interpreter(f"{output_path}")
    return optimized_scripted_module


def _convert_rx101_to_mobile(model_path: str, num_classes: int,
                             output_path: str or Path = "./rx101_optimized_scripted1.ptl"):
    """Function handling conversion of MSCG-Net Rx101 Models

    :param model_path:
    :param num_classes:
    :param output_path:
    :return:
    """
    # PATH='/content/drive/MyDrive/MSCGNET_pradeep_clone/P3-SemanticSegmentation-main/ckpt/epoch_8_loss_0.99527_acc_0.82278_acc-cls_0.60967_mean-iu_0.48098_fwavacc_0.70248_f1_0.62839_lr_0.0000829109.pth'
    PATH = '/content/drive/MyDrive/MSCGNET_pradeep_clone/P3-SemanticSegmentation-main/ckpt/MSCG-Rx50-epoch_4_loss_1.12325_acc_0.76337_acc-cls_0.52928_mean-iu_0.39289_fwavacc_0.62579_f1_0.53039_lr_0.0000856692.pth'
    # metrics = torch.load(PATH,map_location=torch.device('cpu'))

    model = RX101GCN3Head4Channel()
    for param in model.parameters():
        param.requires_grad = False

    print(model.graph_layers2.fc[0])
    # Add on classifier for the number of classes
    model.graph_layers2.fc[0] = nn.Linear(128, 10)
    # print(metrics)
    model.scg.mu[0] = nn.Conv2d(1024, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.scg.logvar[0] = nn.Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')), strict=False)
    model.eval()
    print(model.eval())
    # print(summary(metrics,(10,4,32,32)))
    example = torch.rand(10, 4, 32, 32)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("rx101_jit_traced.pt")

    # normal .ptl file
    traced_script_module._save_for_lite_interpreter("rx101_scripted1.ptl")

    optimized_scripted_module = optimize_for_mobile(traced_script_module)
    # optimized .ptl file
    optimized_scripted_module._save_for_lite_interpreter(f"{output_path}")
    return optimized_scripted_module


if __name__ == "__main__":
    checkpoint_path = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/adam/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/MSCG-Rx50-epoch_2_loss_1.17060_acc_0.76049_acc-cls_0.47790_mean-iu_0.35847_fwavacc_0.63065_f1_0.49588_lr_0.0000863686.pth"
    _convert_rx50_to_mobile(checkpoint_path)
    # metrics = torchvision.core.mobilenet_v2(pretrained=True)
    pass
