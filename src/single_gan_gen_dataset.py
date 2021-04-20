import os
import numpy as np

import torch

from tqdm import tqdm
from datetime import datetime

from torchvision.utils import save_image

from experiment import ExperimentLog
from utils import encode_onehot

EXPERIMENT_ID = input("Experiment ID: ")
EXPERIMENT_PATH = f'logs/cifar10/single_gan/experiment_{EXPERIMENT_ID}'

# Get the experiment log
LOGGER = ExperimentLog(f"{EXPERIMENT_PATH}/log_generate_dataset")

# Get best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Parameters
CLASSES = 10
TRAINING_SIZE = 900
MULTIPLIER_TRAINING_SIZE = 10
FAKESET = f"data/vgg16/dataset_gan{EXPERIMENT_ID}/"
if not os.path.exists(FAKESET):
    os.mkdir(FAKESET)

LOGGER.write(f'#### Experiment {EXPERIMENT_ID} ####')
LOGGER.write(f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}')

LOGGER.write('\nHiperparametros')
LOGGER.write(f'> Classes: {CLASSES}')
LOGGER.write(f'> Training Size: {TRAINING_SIZE}*{MULTIPLIER_TRAINING_SIZE}')
LOGGER.write(f'> Device: {DEVICE}')

LOGGER.write(f"\nLoading generator")
g_model = torch.load(f'models/gan/g_exp{EXPERIMENT_ID}.pth')

def generate_conditional_latent_points(latent_dim, n_samples, n_class, device):
    noise = torch.randn(n_samples, latent_dim).to(device)
    labels = torch.randint(0, CLASSES, (n_samples, 1))
    labels.zero_()
    labels.fill_(n_class)

    conditions = encode_onehot(
        labels,
        CLASSES
    ).to(device)

    return noise, conditions 

for n in np.arange(CLASSES):
    LOGGER.write(f"Generating images from class {n}")
    output_path = os.path.join(FAKESET, str(n))
    LOGGER.write(f'> Output path: {output_path}')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in tqdm(np.arange(MULTIPLIER_TRAINING_SIZE)):
        X, y = generate_conditional_latent_points(100, TRAINING_SIZE, n, DEVICE)

        generated = g_model(X, y)

        for id, image in enumerate(generated):
            new_id = (TRAINING_SIZE * i) + id
            save_image(
                (image + 1) / 2, # normalize into [0, 1]
                fp=os.path.join(output_path, f"{new_id}.png")
            )
