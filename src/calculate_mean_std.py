import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse

import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Copycat Training')
parser.add_argument('--path', type=str, help='path of folder to train the copycat')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor()
])

# trainset = torchvision.datasets.MNIST(
    # root='./data', train=True, download=True, transform=transform)
# trainset = torchvision.datasets.EMNIST(
    # root='./data', split="digits", download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=False, num_workers=4)

pop_mean = []
pop_std0 = []
pop_std1 = []
for i, data in enumerate(dataloader, 0):
    # shape (batch_size, 3, height, width)
    numpy_image = data[0].numpy()
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))
    batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)
pop_std1 = np.array(pop_std1).mean(axis=0)

print('mean', pop_mean)
print('std', pop_std1)
