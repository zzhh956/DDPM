import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import save_image

class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.path = Path('../mnist')
        self.path_list = sorted(self.path.rglob("*.png"))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = str(self.path_list[index])

        # width x height target sizes: [28, 28, 3]
        img = cv2.imread(path)

        # change color sequence
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #resize
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        #totensor -> batch_size x color(rgb) x width x height 
        img = self.bytetotensor(img)

        return img

    def bytetotensor(self, x):
        transform = T.Compose([
            T.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )

        return transform(x)

class GaussianNoise(Dataset):
    def __init__(self, input_shape, length = 10000):
        super().__init__()
        self.shape = input_shape
        self.length = length
        self.list_of_noise = [torch.randn(self.shape) for _ in range(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.list_of_noise[index]