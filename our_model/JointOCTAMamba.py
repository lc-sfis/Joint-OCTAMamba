# filename: MultiTaskOCTAMamba_FARGO_Interactive.py 
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

try:
    from .RVMamba import    RVMamba as RV_Model
    from .FAZMamba import FAZMamba as FAZ_Model
except ImportError:
    from our_model.RVMamba import RVMamba as RV_Model
    from our_model.FAZMamba import  FAZMamba as FAZ_Model
#torch.autograd.set_detect_anomaly(True)
def center_crop_tensor(tensor: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
    _, _, h, w = tensor.shape
    th, tw = crop_size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return tensor[:, :, y1:y1 + th, x1:x1 + tw]

class RVPriorMamba(nn.Module):
    def __init__(self, tasks: List[str], use_checkpoint: bool = True, faz_crop_size: int = 128, end_to_end: bool = False):
        super().__init__()
        print("🚀 Initializing FARGO-style INTERACTIVE model with Train/Eval FAZ logic.")
        
        # self.tasks = tasks
        # self.faz_crop_size = (faz_crop_size, faz_crop_size)

        # self.rv_model = RV_Model()
        # self.faz_model = FAZ_Model()
        
        # original_conv = self.faz_model.qseme.init_conv[0]
        # new_in_channels = 2
        self.tasks = tasks
        self.faz_crop_size = (faz_crop_size, faz_crop_size)
        self.end_to_end = end_to_end # 
        
        mode = "END-TO-END" if self.end_to_end else "DETACHED"
        print(f"🚀 Initializing Model 1 (End-to-End Control): Mode set to [ {mode} ]")
        self.rv_model = RV_Model()
        self.faz_model = FAZ_Model()
        
        original_conv = self.faz_model.qseme.init_conv[0]
        new_in_channels = 2
        if original_conv.in_channels != new_in_channels:
            new_conv = nn.Conv2d(
                in_channels=new_in_channels, out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size, stride=original_conv.stride,
                padding=original_conv.padding, bias=(original_conv.bias is not None)
            )
            with torch.no_grad():
                original_weights = original_conv.weight.clone()
                new_conv.weight.zero_()
                new_conv.weight[:, 0:1, :, :] = original_weights
                if original_conv.bias is not None:
                    new_conv.bias.data.copy_(original_conv.bias.data)
            self.faz_model.qseme.init_conv[0] = new_conv
            print(f"   - FAZ model's input layer modified for {new_in_channels} channels.")
    def forward(self, x: torch.Tensor, task: str = "OCTA500_3M") -> Dict[str, torch.Tensor]:
        rv_output = self.rv_model(x)
        if self.end_to_end:
            rv_for_faz_input = rv_output
        else:
            rv_for_faz_input = rv_output.detach()
        rv_for_faz_input = rv_for_faz_input.clamp(0.01, 0.99)
        faz_global_input = torch.cat([x, rv_for_faz_input], dim=1)
        faz_cropped_input = center_crop_tensor(faz_global_input, self.faz_crop_size)
        faz_cropped_output = self.faz_model(faz_cropped_input)
        if self.training:
            return {"rv": rv_output, "faz_cropped": faz_cropped_output}
        else:
            _, _, h, w = x.shape; th, tw = self.faz_crop_size
            x1 = int(round((w - tw) / 2.)); y1 = int(round((h - th) / 2.))
            faz_full_output = torch.zeros_like(x[:, :1, :, :])
            faz_full_output[:, :, y1:y1 + th, x1:x1 + tw] = faz_cropped_output
            return {"rv": rv_output, "faz": faz_full_output}






