from utils.utils import save_images, plot_images, setup_logging
from model.blocks import UNet_conditional, EMA
from model.diffusion_model import Diffusion

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np

from tqdm import tqdm
import logging
import copy
import os


def train(args, data):

    # Create saving directories
    setup_logging(args.name_exp)

    device = args.device
    dataloader = data.train
    model = UNet_conditional(num_classes=data.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=data.img_size, device=device)

    logger = SummaryWriter(os.path.join("runs", args.name_exp))
    size_dataset = len(dataloader)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):

        logging.info(f"Starting epoch {epoch}:")
        process_bar = tqdm(dataloader)

        for i, (images, labels) in enumerate(process_bar):

            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if np.random.random() < 0.1:
                labels = None

            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            process_bar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * size_dataset + i)

        if epoch % 10 == 0:

            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

            # Save diffusion process images
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.name_exp, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.name_exp, f"{epoch}_ema.jpg"))

            # Save weights
            torch.save(model.state_dict(), os.path.join("models", args.name_exp, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.name_exp, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.name_exp, f"optim.pt"))

            # TODO : what do I want to save more ?
            #   - The loss
