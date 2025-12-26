import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model.unet import UNet
from diffusion.ddpm import DDPM
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np


def train(
    total_epoch,
    model,
    diffusion,
    train_dataloader,
    test_dataloader,
    lr=1e-4,
    resume=None,
    writer=None,
    module_save_path=None,
):
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    # resume train
    if resume is not None and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, weights_only=True))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(opt, step_size=50, gamma=0.1)
    # print(scheduler.get_last_lr())
    noise_img = torch.randn(9, 3, 32, 32)
    for epoch in range(total_epoch):
        # 训练
        model.train()
        train_losses = []
        eval_losses = []
        for imgs, _ in tqdm(train_dataloader, desc=f"epoch:{epoch}"):
            t = torch.randint(0, diffusion.num_timesteps, (imgs.shape[0],))
            imgs, noise = diffusion.add_noise(imgs, t)
            imgs = imgs.to(torch.float32).to(device)
            noise = noise.to(device)
            output = model(imgs, t)
            loss = F.mse_loss(output, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        # 评估
        model.eval()
        with torch.no_grad():
            for imgs, _ in test_dataloader:
                t = torch.randint(0, diffusion.num_timesteps, (imgs.shape[0],))
                imgs, noise = diffusion.add_noise(imgs, t)
                imgs = imgs.to(torch.float32).to(device)
                noise = noise.to(device)
                output = model(imgs, t)
                loss = F.mse_loss(output, noise)
                eval_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        if module_save_path is not None:
            torch.save(
                model.state_dict(), os.path.join(module_save_path, f"model_ep{epoch}_tl_{avg_eval_loss:0.4f}.pth")
            )

        # denoise
        indices = list(range(1000))[::-1]
        show_imgs = [noise_img]
        denoise_img = noise_img.clone()
        with torch.no_grad():
            model.eval()
            for t in indices:
                denoise_img = diffusion.denoise(model, denoise_img, t, device, True)
                if t % 100 == 0:
                    show_imgs.append(denoise_img.cpu())
        show_imgs = [(torch.clamp(img, min=-1, max=1) + 1) / 2 for img in show_imgs]
        concat_img = [make_grid(denoise_img, nrow=3, padding=2, normalize=False) for denoise_img in show_imgs]
        concat_img = torch.stack(concat_img, dim=0)
        if writer is not None:
            writer.add_scalar("train/loss", avg_train_loss, epoch)
            writer.add_scalar("eval/loss", avg_eval_loss, epoch)
            writer.add_images("eval/generate", concat_img, epoch)
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
        scheduler.step()


def lets_go(input_args):
    random.seed(input_args.seed)  # Python 内置 random 模块
    np.random.seed(input_args.seed)  # NumPy 随机数
    torch.manual_seed(input_args.seed)  # CPU 上的 PyTorch 随机数

    writer = SummaryWriter(f"train_saves/{input_args.tran_id}/runs")

    dataset_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    training_data = datasets.CIFAR100(root="data", train=True, download=True, transform=dataset_transforms)

    test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=dataset_transforms)

    train_dataloader = DataLoader(
        training_data,
        batch_size=input_args.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=input_args.batch_size,
        shuffle=True,
    )
    model = UNet()
    solver = DDPM()
    save_weight_folder = f"train_saves/{input_args.tran_id}/weights"
    if not os.path.exists(save_weight_folder):
        os.makedirs(save_weight_folder)
    train(
        input_args.epoch_size,
        model,
        solver,
        train_dataloader,
        test_dataloader,
        lr=input_args.learning_rate,
        writer=writer,
        module_save_path=save_weight_folder,
    )


def main():
    parser = argparse.ArgumentParser("Toy Diffusion")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch_size", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--tran_id", type=str, default="toy2_sheduler_lr")
    parser.add_argument("--seed", type=int, default=1225)
    lets_go(parser.parse_args())


if __name__ == "__main__":
    main()

#

# train log:
"""
toy2: scheduler learning_rate

"""


#
