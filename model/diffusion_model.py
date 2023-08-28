import logging
import torch

from tqdm import tqdm
from einops import *


class Diffusion:
    def __init__(self, noise_steps=10, beta_start=1e-4, beta_end=0.02, img_size=224, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_alpha_hat = rearrange(sqrt_alpha_hat, 'b -> b 1 1 1')

        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = rearrange(sqrt_one_minus_alpha_hat, 'b -> b 1 1 1')

        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():

            size_random_tensor = (n, 3, self.img_size, self.img_size)
            x = torch.randn(size_random_tensor).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                t = (torch.ones(n) * i).long().to(self.device)
                # Predict the (value ?) of the noise
                predicted_noise = model(x, t, labels)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = rearrange(self.alpha[t], 'b -> b h w c')
                alpha_hat = rearrange(self.alpha_hat[t], 'b -> b h w c')
                beta = rearrange(self.beta[t], 'b -> b h w c')

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # Compute associated previous images (denoising it)
                noise_x_alpha = (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                x = 1 / torch.sqrt(alpha) * noise_x_alpha + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x
