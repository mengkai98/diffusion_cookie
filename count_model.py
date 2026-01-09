import torch
import torch.nn as nn
from model.unet_ddpm import UNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


model = EMA(DDPM_UNET())
total_params = count_parameters(model)
print(total_params)
