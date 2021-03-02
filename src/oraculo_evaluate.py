import os
import sys
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10, STL10, ImageFolder

from experiment import ExperimentLog
from models.vgg import VGG

def evaluate_network(model, dataloader, total):
    correct = 0
    res_pos = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    results = np.zeros([total, 2], dtype=np.int)

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            results[res_pos:res_pos+labels.size(0), :] = np.array([labels.tolist(), predicted.tolist()]).T
            res_pos += labels.size(0)

            correct += (predicted == labels).sum().item()

    micro_avg = f1_score(results[:,0], results[:,1], average='micro')
    macro_avg = f1_score(results[:,0], results[:,1], average='macro')
    print('Average: {:.2f}% ({:d} images)'.format(100. * (correct/total), total))
    print('Micro Average: {:.6f}'.format(micro_avg))
    print('Macro Average: {:.6f}\n'.format(macro_avg))


def evaluate_network_with_classes(model, dataloader, total):
    def micro_avg_class(x):
        results_class = results[np.where(results[:, 0] == x)]
        aux = np.where(results_class[:, 1] == x)
        results_class[:, 1] = 0
        results_class[aux, 1] = x        
        return f"{(100*f1_score(results_class[:,0], results_class[:,1], average='micro')):.2f}%"
    def macro_avg_class(x):
        results_class = results[np.where(results[:, 0] == x)]
        aux = np.where(results_class[:, 1] == x)
        results_class[:, 1] = 0
        results_class[aux, 1] = x
        return f"{(100*f1_score(results_class[:,0], results_class[:,1], average='macro')):.2f}%"

    correct = 0
    res_pos = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    results = np.zeros([total, 2], dtype=np.int)

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            results[res_pos:res_pos+labels.size(0), :] = np.array([labels.tolist(), predicted.tolist()]).T
            res_pos += labels.size(0)

            correct += (predicted == labels).sum().item()

    micro_avg = f1_score(results[:,0], results[:,1], average='micro')
    macro_avg = f1_score(results[:,0], results[:,1], average='macro')
    print('Average: {:.2f}% ({:d} images)'.format(100. * (correct/total), total))
    print(
        'Micro Average: {:.2f}% ['.format(100 * micro_avg),
        ", ".join(map(micro_avg_class, np.arange(10))), "]", sep=""
    )
    print(
        'Macro Average: {:.2f}% ['.format(100 * macro_avg),
        ", ".join(map(macro_avg_class, np.arange(10))), "]\n", sep=""
    )

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

    evaluate_network(net, testloader, len(testset))

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

target = torch.load('models/target/cifar10.vgg19.pth')
target = target.to(DEVICE)

#evaluate_cifar10(target)
#evaluate_stl10(target)
#evaluate_dataset(target, 'data/dataset_i')
#evaluate_dataset(target, 'data/dataset_ii')
evaluate_dataset(target, 'data/dataset_gan113')
evaluate_dataset(target, 'data/dataset_gan114')