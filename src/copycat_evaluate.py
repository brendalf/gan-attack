import argparse

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
    # Normalize,
    # RandomCrop,
    # RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from torchvision.transforms.transforms import Grayscale

from evaluate import evaluate_network_with_classes
from models.vgg import VGG16

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

def evaluate_svhn(net):
    testset = SVHN(
        root='data', 
        split="test", 
        download=True, 
        transform=Compose([
            ToTensor(),
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
        ])
    )

    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader, len(testset), OUT_FEATURES)

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Copycat Training')
parser.add_argument('--path', type=str, help='path of folder to train the copycat')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = args.path
OUT_FEATURES=10

target = VGG16(out_features=OUT_FEATURES)
target = target.to(DEVICE)
checkpoint = torch.load(MODEL_PATH)

target = nn.DataParallel(target)
cudnn.benchmark = True
target.load_state_dict(checkpoint['net'])
print("Epoch: ", checkpoint['epoch'] + 1)

#evaluate_dataset(target, 'data/FER7/TD', OUT_FEATURES)
#evaluate_cifar10(target)
# evaluate_svhn(target)
evaluate_mnist(target)
