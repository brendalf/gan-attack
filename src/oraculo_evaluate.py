import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10, STL10, ImageFolder


from experiment import ExperimentLog
from models.vgg import VGG16
from evaluate import evaluate_network, evaluate_network_with_classes

def evaluate_cifar10(net):
    testset = CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset))

def evaluate_stl10(net):
    testset = STL10(
        root='data',
        split='train',
        download=True,
        transform=Compose([
            Resize((32,32)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset))

def evaluate_dataset(net, path):
    testset = ImageFolder(
        root=path,
        transform=Compose([
            Resize((32,32)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset))

np.set_printoptions(suppress=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/target/cifar10.vgg16.pth'

checkpoint = torch.load(MODEL_PATH)
if isinstance(checkpoint, dict):
    target = VGG16()
    target = target.to(DEVICE)
    checkpoint = torch.load(MODEL_PATH)

    target = torch.nn.DataParallel(target)
    cudnn.benchmark = True
    target.load_state_dict(checkpoint['net'])
    print("Epoch: ", checkpoint['epoch'] + 1)
else:
    target = torch.load(MODEL_PATH)
    target.to(DEVICE)

evaluate_cifar10(target)
evaluate_stl10(target)
#evaluate_dataset(target, 'data/dataset_i')
#evaluate_dataset(target, 'data/dataset_ii')
#evaluate_dataset(target, 'data/dataset_gan113')
#evaluate_dataset(target, 'data/dataset_gan114')
