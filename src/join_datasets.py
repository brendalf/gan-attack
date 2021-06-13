import os
import torch
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(suppress=True)

dataset1_folder = 'data/mnist.vgg16/dataset_iv_sl'
dataset2_folder = 'data/mnist.vgg16/dataset_gan183_sl'

new_folder = "data/mnist.vgg16/dataset_iv_sl+dataset_gan183_sl"

dataset1 = ImageFolder(
    root=dataset1_folder,
    transform=Compose([
        # Resize((32, 32)),
        ToTensor(),
    ])
)
loader1 = DataLoader(dataset1, shuffle=True, batch_size=1)

dataset2 = ImageFolder(
    root=dataset2_folder, 
    transform=Compose([
        # Resize((32, 32)),
        ToTensor(),
    ])
)
loader2 = DataLoader(dataset2, shuffle=True, batch_size=1)

os.makedirs(new_folder, exist_ok=True)

labels = {k:0 for k in np.arange(10)}


for label in labels.keys():
    label_folder = os.path.join(new_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

print('Joining datasets...')
for image, label in tqdm(loader1):
    label = label.item()
    label_folder = os.path.join(new_folder, str(label))

    image = image.to(DEVICE)

    i = labels[label]
    labels[label] += 1

    save_image(
        image,
        fp=os.path.join(label_folder, f'{i}.png')
    )


for image, label in tqdm(loader2):
    label = label.item()
    label_folder = os.path.join(new_folder, str(label))

    image = image.to(DEVICE)

    i = labels[label]
    labels[label] += 1

    save_image(
        image,
        fp=os.path.join(label_folder, f'{i}.png')
    )
