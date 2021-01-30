import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10, ImageFolder

#from utils import encodeOneHot
from experiment import ExperimentLog
from models.generator import GeneratorDCGAN as Generator
from models.vgg19 import VGG19

from utils import generate_latent_points


EXPERIMENT_ID = 65
EXPERIMENT_PATH = f'logs/experiments/experiment_{EXPERIMENT_ID}'

# Get the experiment log
LOGGER = ExperimentLog(f"{EXPERIMENT_PATH}/log_copycat")

# Get best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Parameters
EPOCHS = 50
WORKERS = 2
CLASSES = 10
BATCH_SIZE = 32
TRAINING_SIZE = 1000
FAKESET = "data/copycat_generated_dataset/"
if not os.path.exists(FAKESET):
    os.mkdir(FAKESET)

LOGGER.write(f'#### Experiment {EXPERIMENT_ID} ####')
LOGGER.write(f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}')

LOGGER.write('\nHiperparametros')
LOGGER.write(f'> Epochs: {EPOCHS}')
LOGGER.write(f'> Classes: {CLASSES}')
LOGGER.write(f'> Batch Size: {BATCH_SIZE}')
LOGGER.write(f'> Training Size: {TRAINING_SIZE}*10')
LOGGER.write(f'> Device: {DEVICE}')

for n in np.arange(0, CLASSES):
    LOGGER.write(f"\nLoading generator from class {n}")
    g_model = torch.load(f'models/adversary/g_exp{EXPERIMENT_ID}_class{n}.pth')

    LOGGER.write(f"Generating images from class {n}")
    output_path = os.path.join(FAKESET, str(n))
    LOGGER.write(f'> Output path: {output_path}')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in np.arange(0, 100):
        X = generate_latent_points(100, TRAINING_SIZE, DEVICE)

        generated = g_model(X)

        for id, image in enumerate(generated):
            new_id = (TRAINING_SIZE * i) + id
            save_image(
                image,
                fp=os.path.join(output_path, f"{new_id}.png")
            )

LOGGER.write("Loading generated dataset")
imagefolder = ImageFolder(
    FAKESET,
    transform=Compose([
        Resize((32,32)),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)
dataloader = DataLoader(
    imagefolder,
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    shuffle=True
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
        Resize((32,32)),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

LOGGER.write("Starting copycat training")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
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

    acc = (100 * correct / total)
    LOGGER.write(f'E: {epoch+1}/{EPOCHS} [train_loss: {running_loss:.3f}, val_loss: {val_loss:.3f}, val_acc: {acc:.3f}]')