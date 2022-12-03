import os
import ipdb
import torch
import random
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from model import UNet
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_gan_metrics import get_fid
from dataset import MNIST, GaussianNoise
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from torch.utils.data.sampler import SubsetRandomSampler

class DiffusionTrainer(pl.LightningModule):
    def __init__(self, grid=True):
        super().__init__()
        os.makedirs('../images', exist_ok=True)
        self.train_total = 0
        self.train_loss = 0
        self.grid = grid

        self.T = 1000
        self.beta = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32).view(-1, 1, 1, 1).to(device='cuda') # all hyper-parameters
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device='cuda')
        self.sigma = torch.sqrt(self.beta).to(device='cuda')

        # model
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.model = UNet(input_shape=(3, 32, 32), time_step=self.T, ch=64, num_res_blocks=2, ch_mults=[1, 2, 2, 2], attn_res=[16, 8, 4])

    def prepare_data(self):
        self.train_dataset = MNIST()
        self.train_idx = list(range(len(self.train_dataset)))

        self.grid_dataset = GaussianNoise((3, 32, 32), length=1)
        self.sample_dataset = GaussianNoise((3, 32, 32), length=10000)

    def train_dataloader(self):
        sampler = SubsetRandomSampler(self.train_idx)

        return DataLoader(self.train_dataset, batch_size = self.batch_size, sampler = sampler, num_workers = torch.get_num_threads(), pin_memory = True)

    def test_dataloader(self):
        if self.grid is True:
            return DataLoader(self.grid_dataset, batch_size = 1, pin_memory = True)
        else:
            return DataLoader(self.sample_dataset, batch_size = self.batch_size, num_workers = torch.get_num_threads(), pin_memory = True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.learning_rate)

    def training_step(self, batch, batch_idx):
        x0 = batch
        times = torch.randint(self.T, size=(x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0, device=x0.device)

        xt = torch.sqrt(self.alpha_bar[times]) * x0 + torch.sqrt(1.0 - self.alpha_bar[times]) * noise
        y = self.model(xt, times)
        
        loss = F.mse_loss(y, noise)

        self.train_loss += loss.item()
        self.train_total += 1
        self.log('avg_loss', self.train_loss/self.train_total, prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        self.train_loss = 0
        self.train_total = 0
    
    def test_step(self, batch, batch_idx):
        xt = batch

        if self.grid:
            idx = torch.linspace(0, self.T, 8, dtype=torch.long, device='cuda')[:-1]
            image = torch.tensor(1)

            for i in range(8):
                image_col = [torch.ones_like(torch.empty(1, 3, 28, 28), device='cuda')]
                xt = batch

                for t in reversed(torch.arange(self.T, device=xt.device)):
                    z = torch.randn_like(xt, device=xt.device) if t > 0 else 0
                    xt = self.predict_mean(xt, t) + self.sigma[t] * z

                    if t in idx:
                        image_col.append(resize(torch.clamp(xt, min=0., max=1.), [28, 28]))

                image_col = torch.stack(image_col)

                if i == 0:
                    image = image_col
                else:
                    image = torch.cat((image, image_col), dim=1)

            image = torch.permute(image, (2, 0, 3, 1, 4))
            image = torch.reshape(image, (3, 8 * 28, 8 * 28))
            print(image.shape)

            return image
        else:
            for t in reversed(torch.arange(self.T, device=xt.device)):
                z = torch.randn_like(xt, device=xt.device) if t > 0 else 0
                xt = self.predict_mean(xt, t) + self.sigma[t] * z

            return torch.clamp(xt, min=0., max=1.)

    def test_epoch_end(self, outputs):
        if self.grid:
            print(len(outputs))
            path = os.path.join(f'../311513015.png')
            save_image(outputs, path)
        else:
            i = 1

            for output in outputs:
                for img in output:
                    img = resize(img, [28, 28])
                    path = os.path.join(f'../images/{i:05d}.png')
                    save_image(img, path)
                    i = i + 1
                
    def predict_mean(self, xt, t) -> torch.Tensor:
        pred_noise = self.model(xt, t)
        xt = xt - self.beta[t] / torch.sqrt(1.0 - self.alpha_bar[t]) * pred_noise
        xt = xt / torch.sqrt(self.alpha[t])

        return xt