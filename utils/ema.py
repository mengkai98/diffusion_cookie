import copy
import torch.nn as nn
import torch


class EMA(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.main_model: nn.Module = model
        self.ema_model: nn.Module = copy.deepcopy(model)

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def state_dict(self):
        return {
            "model": self.main_model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.main_model.load_state_dict(state_dict["model"], strict=strict)
        self.ema_model.load_state_dict(state_dict["ema"], strict=strict)

    def forward(self, *args, **kwargs):

        if self.training:
            return self.main_model(*args, **kwargs)
        else:
            return self.ema_model(*args, **kwargs)

    def update_ema(self, beta=0.999):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.state_dict().values(), self.main_model.state_dict().values()):
                ema_v.copy_(beta * ema_v + (1.0 - beta) * model_v)

    def train(self, mode=True):
        super().train(mode)
        self.main_model.train(mode)
        self.ema_model.train(False)
        return self

    def eval(self):
        return self.train(False)


#
#
