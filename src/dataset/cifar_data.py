import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
    ])
    
    return transform


def get_trainset(root='./data', batch=4):
    ret = {}
    transform = get_transform()

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2
    )
    ret['loader'] = trainloader

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    ret['classes'] = classes
    
    return ret


def get_testset(root='./data', batch=4):
    ret = {}
    transform = get_transform()
    
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=2
    )
    ret['loader'] = testloader

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    ret['classes'] = classes
    
    return ret