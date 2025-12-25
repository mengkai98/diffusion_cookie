import numpy as np
import torch


class DDPM:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000):
        self.betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.num_timesteps = int(self.betas.shape[0])
        self.fix_noise = {}

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.from_numpy(self.sqrt_alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod[t])
        noise_image = (
            sqrt_alphas_cumprod.view(-1, 1, 1, 1) * x + sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1) * noise
        )
        return noise_image, noise

    def denoise(self, model, x, t, device=torch.device("cpu"), use_fix_noise=False):
        alpha = self.alphas[t]
        sqrt_alpha = np.sqrt(alpha)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        beta = self.betas[t]
        inpu_t = torch.full((x.shape[0],), t).to(device)
        x = x.to(device)
        pred_noise = model(x, inpu_t)
        x = (x - (1 - alpha) / sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha
        if t > 0:
            if use_fix_noise:
                if t not in self.fix_noise:
                    self.fix_noise[t] = torch.randn_like(x).to(device)
                noise = self.fix_noise[t]
            else:
                noise = torch.randn_like(x).to(device)
            x = x + np.sqrt(beta) * noise
        return x


#
#
