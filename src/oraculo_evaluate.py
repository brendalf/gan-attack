import os
import sys
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10, STL10, ImageFolder

from experiment import ExperimentLog
from models.vgg import VGG

from oraculo.train import train_network

def evaluate_network(model, dataloader):
    """
    Evaluate the trained model with the testset inside dataloader
    INPUT
        model: the trained network pytorch model
        dataloader: the test set dataloader
    OUTPUT
        accuracy: network accuracy
    """
    correct = 0
    total = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            #predicted = predicted.view(labels.size(0), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Accuracy: {acc}%')
    return acc

def evaluate_network_with_classes(model, dataloader):
    """
    Evaluate the trained model with the testset inside dataloader
    INPUT
        model: the trained network pytorch model
        dataloader: the test set dataloader
    OUTPUT
        accuracy: network accuracy
    """
    def perc(x):
        return f"{x}: {(pred_dict[x]/total_dict[x]):.2f}%"

    correct = 0
    total = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    pred_dict = {x: 0 for x in range(10)}
    total_dict = {x: 0 for x in range(10)}

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().item()
            labels = labels.cpu().item()

            total += 1
            total_dict[labels] += 1

            if predicted == labels:
                correct += 1
                pred_dict[predicted] += 1

    acc = 100 * correct / total
    print(f'Accuracy: {acc}% [', ", ".join(map(perc, np.unique(list(pred_dict.keys())))), "]", sep="")
    return acc

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
        testset, batch_size=32, shuffle=False, num_workers=2
    )

    evaluate_network(net, testloader)

def evaluate_stl10(net):
    testset = STL10(
        root='data', 
        split='test', 
        download=True, 
        transform=Compose([
            Resize((32,32)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2
    )

    evaluate_network(net, testloader)

def evaluate_attack(net):
    testset = ImageFolder(
        root='data/cifar10_attack_v2/',
        transform=Compose([
            Resize((32,32)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )

    testloader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader)

def evaluate_gan_dataset(net):
    testset = ImageFolder(
        root='data/copycat_generated_dataset/',
        transform=Compose([
            Resize((32,32)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )

    testloader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2
    )

    evaluate_network_with_classes(net, testloader)

np.set_printoptions(suppress=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

target = torch.load('models/target/cifar10.vgg19.pth')
target = target.to(DEVICE)

evaluate_cifar10(target)
evaluate_stl10(target)
evaluate_attack(target)
evaluate_gan_dataset(target)