import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchsummary import summary
from models.mscg import get_model

if __name__ == "__main__":
    # model = torchvision.models.mobilenet_v2(pretrained=True)

    model = get_model(
        name="MSCG-Rx101", classes=10,
        node_size=(32, 32)
    )
    filepath = "/home/hanz/github/P3-SemanticSegmentation/checkpoints/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/MSCG-Rx101-epoch_4_loss_1.64719_acc_0.73877_acc-cls_0.47181_mean-iu_0.34377_fwavacc_0.59123_f1_0.48653_lr_0.0001245001.pth"
    model = model.load_state_dict(torch.load(filepath,map_location=torch.device('cpu')))
    input_size = (
        3, 512, 512
    )

    # model = torch.load(filepath)
    #    print(model)
    #     print(summary(model, input_size=input_size))
    #     model.eval()
    # example = torch.rand(1, 3, 512,512)
    traced_script_module = torch.jit.trace(model, input_size)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("./model.ptl")
