import argparse
import os
from utils import InvertTransform
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, STL10
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.svhn import SVHN
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from torchvision.transforms.transforms import Grayscale
from tqdm import tqdm

from evaluate import evaluate_network, evaluate_network_with_classes
from experiment import ExperimentLog
from models.vgg import VGG16

def evaluate_cifar10(net):
    testset = CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset), OUT_FEATURES)

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

    evaluate_network_with_classes(net, testloader, len(testset), OUT_FEATURES)

def evaluate_svhn(net):
    testset = SVHN(
        root='data', 
        split="train", 
        download=True, 
        transform=Compose([
            ToTensor(),
            Normalize((0.43767047, 0.44375867, 0.47279018), (0.19798356, 0.20096427, 0.19697163)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset), OUT_FEATURES)

def evaluate_mnist(net):
    testset = MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=Compose([
            Resize((32,32)),
            Grayscale(num_output_channels=3),
            ToTensor(),
            Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset), OUT_FEATURES)

def evaluate_dataset(net, path):
    testset = ImageFolder(
        root=path,
        transform=Compose([
            Resize((32,32)),
            ToTensor(),
            Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),     
            # Normalize((0.43767047, 0.44375867, 0.47279018), (0.19798356, 0.20096427, 0.19697163)),     
            # Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset), OUT_FEATURES)

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Copycat Training')
parser.add_argument('--name', type=str, help='Name of model')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = args.name
OUT_FEATURES = 10

checkpoint = torch.load(f"models/target/{MODEL_NAME}.pth")
if isinstance(checkpoint, dict):
    target = VGG16(out_features=OUT_FEATURES)
    target = target.to(DEVICE)

    target = nn.DataParallel(target)
    cudnn.benchmark = True
    target.load_state_dict(checkpoint['net'])
    print("Epoch: ", checkpoint['epoch'] + 1)
else:
    target = torch.load(f"models/target/{MODEL_NAME}.pth")
    target.to(DEVICE)

evaluate_cifar10(target)
# evaluate_stl10(target)
# evaluate_svhn(target)
# evaluate_mnist(target)
# evaluate_dataset(target, 'data/mnist_pd_ol_1000')
