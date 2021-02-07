import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10, ImageFolder

from experiment import ExperimentLog
from models.vgg import VGG19


EXPERIMENT_ID = input("Experiment ID: ")
EXPERIMENT_PATH = f'logs/experiments/experiment_{EXPERIMENT_ID}'

# Get the experiment log
LOGGER = ExperimentLog(f"{EXPERIMENT_PATH}/log_copycat")

# Get best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Parameters
EPOCHS = 30
WORKERS = 4
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 32

#FAKESET = "data/cifar10_attack_v2/"
#FAKESET = "data/copycat_generated_dataset/"
FAKESET = "data/copycat_generated_dataset_stolen_labels/"

if not os.path.exists(FAKESET):
    os.mkdir(FAKESET)

LOGGER.write(f'#### Experiment {EXPERIMENT_ID} ####')
LOGGER.write(f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}')

LOGGER.write('\nHiperparametros')
LOGGER.write(f'> Epochs: {EPOCHS}')
LOGGER.write(f'> Batch Size Train: {BATCH_SIZE_TRAIN}')
LOGGER.write(f'> Batch Size Val: {BATCH_SIZE_VAL}')
LOGGER.write(f'> Device: {DEVICE}')

sw = SummaryWriter(EXPERIMENT_PATH)

LOGGER.write("Loading generated dataset")
imagefolder = ImageFolder(
    FAKESET,
    transform=Compose([
        ToTensor(),
        #Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)
dataloader = DataLoader(
    imagefolder,
    batch_size=BATCH_SIZE_TRAIN,
    num_workers=WORKERS,
    shuffle=True,
    drop_last=True
)

LOGGER.write("\nCopyCat Architecture")
model = VGG19().to(DEVICE)
LOGGER.write(model)

LOGGER.write("Loading testset")
testset = CIFAR10(
    root='../data',
    train=False,
    download=True,
    transform=Compose([
        ToTensor(),
        #Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)
testloader = DataLoader(testset, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=WORKERS)

LOGGER.write("Starting copycat training")
criterion = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=0.0005, amsgrad=True)
lrdecay = ExponentialLR(optimizer=optim, gamma=0.99)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optim.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optim.step()

        train_loss += loss.item()
    else:
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            model.eval()

            for inputs, labels in tqdm(testloader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    lrdecay.step()

    acc = 100 * correct / total
    LOGGER.write(f'E: {epoch+1}/{EPOCHS} [train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, val_acc: {acc:.3f}]')

    sw.add_scalar(f'CopyCat/Train Loss', train_loss, epoch+1)
    sw.add_scalar(f'CopyCat/Val Loss', val_loss, epoch+1)
    sw.add_scalar(f'CopyCat/Val Acc', acc, epoch+1)
    sw.add_scalar(f'CopyCat/LR', lrdecay.get_last_lr()[-1], epoch+1)