from math import ceil

import torch
from torchvision.datasets import ImageNet
from torchvision.transforms import Resize, Compose, RandomHorizontalFlip, RandomResizedCrop, ColorJitter, CenterCrop, ToTensor
from torch.utils.data import DataLoader, Dataset
import math


def make_loader(path, split, batch_size, num_workers):
    if split == 'train':
        transform = Compose([RandomResizedCrop(224,scale=(0.3,4.0/3)), ColorJitter(0.4,0.4,0.4), RandomHorizontalFlip(),
                             ToTensor()])
    else:
        transform = Compose([Resize(256), CenterCrop(224), ToTensor()])


    ds = ImageNet(path, split=split, transform=transform)

    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)

    return loader
