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
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
from utils.ema import EMA
from utils.img_transforms import CustomResizeCrop
from dataset.pet_finder_dataset import PetFinderDataset
import json
from model.unet_ddpm import UNet as DDPM_UNET


class Trainer:
    def __init__(self, input_args) -> None:
        self.parser_args(input_args)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._init_seed(self.seed)
        self._init_noise_img(self.image_size)
        self._init_diffusion()
        self._init_model()
        self._init_dataset(self.dataset, self.image_size, self.batch_size)
        self._init_optimizer(self.lr)
        self._init_saves(self.train_id)
        self.load_weight(self.resume)
        self._save_args(input_args)

    def load_weight(self, resume):
        self.start_epoch = 0
        if resume is not None and os.path.exists(resume):
            save_dict = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(save_dict["model"])
            self.optimizer.load_state_dict(save_dict["optimizer"])
            self.scheduler.load_state_dict(save_dict["sheduler"])
            self.start_epoch = save_dict["epoch"] + 1

    def parser_args(self, input_args):
        self.seed = input_args.seed
        self.resume = input_args.resume
        self.lr = input_args.learning_rate
        self.train_id = input_args.train_id
        self.epoch_size = input_args.epoch_size
        self.batch_size = input_args.batch_size
        self.image_size = input_args.image_size
        self.dataset = input_args.dataset

    def _save_args(self, input_args):
        args_dict = vars(input_args)
        with open(f"train_saves/{input_args.train_id}/args.json", "w") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)

    def _init_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _init_noise_img(self, img_size):
        self.noise_img = torch.randn(9, 3, img_size, img_size)

    def _init_diffusion(self):
        self.diffusion = DDPM()

    def _init_model(self):
        self.model = EMA(DDPM_UNET())

    def _init_optimizer(self, lr=1e-4):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)
        self.scheduler = StepLR(self.optimizer, step_size=80, gamma=0.5)

    def _init_saves(self, train_id):
        self.save_weight_folder = f"train_saves/{train_id}/weights"
        self.denoise_image_folder = f"train_saves/{train_id}/denoise"

        if not os.path.exists(self.save_weight_folder):
            print(f"create {self.save_weight_folder}")
            os.makedirs(self.save_weight_folder)

        if not os.path.exists(self.denoise_image_folder):
            os.makedirs(self.denoise_image_folder)

        self.writer = SummaryWriter(f"train_saves/{train_id}/runs")

    def _init_dataset(self, dataset, img_size, batch_size=32):
        dataset_transforms = transforms.Compose(
            [
                CustomResizeCrop((img_size, img_size)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if dataset == "CIFAR100":
            self.train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=dataset_transforms)
            self.test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=dataset_transforms)

        elif dataset == "PetFinderDataset":
            self.train_data = PetFinderDataset("data/petfinder-pawpularity-score/train.csv", True, dataset_transforms)
            self.test_data = PetFinderDataset("data/petfinder-pawpularity-score/test.csv", False, dataset_transforms)

        else:
            raise RuntimeError()

        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    def train(self):
        device = self.device
        model = self.model
        model.to(device)
        for epoch in range(self.start_epoch, self.epoch_size):
            # 训练
            model.train()
            train_losses = []
            eval_losses = []
            for imgs, _ in tqdm(self.train_dataloader, desc=f"epoch:{epoch}"):
                t = torch.randint(0, self.diffusion.num_timesteps, (imgs.shape[0],), device=device)
                imgs, noise = self.diffusion.add_noise(imgs, t)
                imgs = imgs.to(torch.float32).to(device)
                noise = noise.to(device)
                output = model(imgs, t)
                loss = F.mse_loss(output, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            # 更新EMA模型
            ema_decay = min(0.9999, (1 + epoch) / (10 + epoch))
            model.update_ema(ema_decay)

            # 评估
            model.eval()
            with torch.no_grad():
                for imgs, _ in self.test_dataloader:
                    t = torch.randint(0, self.diffusion.num_timesteps, (imgs.shape[0],), device=device)
                    imgs, noise = self.diffusion.add_noise(imgs, t)
                    imgs = imgs.to(torch.float32).to(device)
                    noise = noise.to(device)
                    output = model(imgs, t)
                    loss = F.mse_loss(output, noise)
                    eval_losses.append(loss.item())
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_eval_loss = sum(eval_losses) / len(eval_losses)

            # denoise
            indices = list(range(1000))[::-1]
            denoise_img = self.noise_img.clone()

            show_imgs = [denoise_img]
            with torch.no_grad():
                model.eval()
                for t in indices:
                    denoise_img = self.diffusion.denoise(self.model, denoise_img, t, device, True)
                    if t % 100 == 0:
                        show_imgs.append(denoise_img.cpu())
            show_imgs = [(torch.clamp(img, min=-1, max=1) + 1) / 2 for img in show_imgs]
            concat_img = [make_grid(denoise_img, nrow=3, padding=2, normalize=False) for denoise_img in show_imgs]
            concat_img = torch.stack(concat_img, dim=0)
            if self.writer is not None:
                self.writer.add_scalar("train/loss", avg_train_loss, epoch)
                self.writer.add_scalar("eval/loss", avg_eval_loss, epoch)
                self.writer.add_scalar("eval/ema_decay", ema_decay, epoch)
                self.writer.add_images("eval/generate", concat_img, epoch)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            save_image(
                concat_img[-1],
                os.path.join(self.denoise_image_folder, f"epoch_{epoch}_.png"),
            )
            self.scheduler.step()
            if self.save_weight_folder is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "sheduler": self.scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.save_weight_folder, f"model_ep{epoch}_tl_{avg_eval_loss:0.4f}.pth"),
                )


def main():
    parser = argparse.ArgumentParser("Toy Diffusion")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch_size", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_id", type=str, default="toy7_new_unet")
    parser.add_argument("--seed", type=int, default=1225)
    parser.add_argument(
        "--resume",
        type=str,
        default="/work/playground/diffusion_cookie/train_saves/toy7_new_unet/weights/model_ep629_tl_0.0374.pth",
    )
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="PetFinderDataset", choices=["CIFAR100", "PetFinderDataset"])
    trainer = Trainer(parser.parse_args())
    trainer.train()


if __name__ == "__main__":
    main()

#

#
