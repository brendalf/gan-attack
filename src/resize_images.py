import os
import torch
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(suppress=True)

dataset_folder = "data/FER7/PD_OL"
new_folder = f"{dataset_folder}_resized"

dataset = ImageFolder(
    root=dataset_folder, 
    transform=Compose([
        Resize((32, 32)),
        ToTensor(),
    ])
)

os.makedirs(new_folder, exist_ok=True)

labels = {k:0 for k in np.arange(7)}

print('Resizing dataset...')
for label in labels.keys():
    idx = np.where(np.array(dataset.targets) == label)[0]
    subset = Subset(dataset, idx)

    loader = DataLoader(subset, shuffle=True, batch_size=1, drop_last=True)

    label_folder = os.path.join(new_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

    for image, _ in loader:
        image = image.to(DEVICE)

        i = labels[label]

        save_image(
            image,
            fp=os.path.join(label_folder, f'{i}.png')
        )

        labels[label] += 1
