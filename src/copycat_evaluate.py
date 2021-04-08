import os
import sys
import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10, STL10, ImageFolder

from models.vgg import VGG16
from evaluate import evaluate_network, evaluate_network_with_classes

def evaluate_cifar10(net):
    testset = CIFAR10(
        root='data', 
        train=False, 
        download=True, 
        transform=Compose([
            ToTensor(),
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
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset))

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Copycat Training')
parser.add_argument('--path', type=str, help='path of folder to train the copycat')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = args.path

target = VGG16()
target = target.to(DEVICE)
checkpoint = torch.load(MODEL_PATH)

target = nn.DataParallel(target)
cudnn.benchmark = True
target.load_state_dict(checkpoint['net'])
print("Epoch: ", checkpoint['epoch'] + 1)

#evaluate_dataset(target, 'data/dataset_gan118_sl_ml')
evaluate_cifar10(target)
