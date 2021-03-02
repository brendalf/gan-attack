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

dataset_folder = 'data/imagenet'
new_folder = 'data/random_images_sl'

dataset = ImageFolder(
    root=dataset_folder, 
    transform=Compose([
    Resize((32, 32)),
    ToTensor(),
]))
loader = DataLoader(dataset, shuffle=True, batch_size=1)

os.makedirs(new_folder, exist_ok=True)

num = 10000

print('Generating random dataset...')
for label in tqdm(np.arange(0, 10)):
    label_folder = os.path.join(new_folder, str(label))

    os.makedirs(label_folder, exist_ok=True)

    for i in tqdm(range(num)):
        image, _ = next(iter(loader))
        image = image.to(DEVICE)

        save_image(
            image,
            fp=os.path.join(label_folder, f'{i}.png')
        )