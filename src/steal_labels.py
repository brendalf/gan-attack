import os
import torch
import numpy as np

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, Normalize, ToTensor

from models.vgg import VGG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(suppress=True)

model = VGG('VGG19')
model = torch.load('models/target/cifar10.vgg19.pth')

norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

dataset_folder = 'data/copycat_generated_dataset'
stolen_labels_folder = 'data/copycat_generated_dataset_stolen_labels'

transform = Compose([
    Resize((32, 32)),
    ToTensor(),
])

dataset = ImageFolder(root=dataset_folder, transform=transform)

if not os.path.exists(stolen_labels_folder):
    os.mkdir(stolen_labels_folder)

from tqdm import tqdm

real_labels = np.array([])
pred_labels = np.array([])

print('Generating labels from target...')
with torch.no_grad():
    model.eval()

    for images, labels in tqdm(dataset):
        real_labels = np.append(real_labels, labels)
        images = images.to(DEVICE)

        # simulando o processamento da api, pois eu n conheco a norm e preciso salvar a imagem original
        images_norm = norm(images.view(3, 32, 32)).view(1, 3, 32, 32)

        outputs = model(images_norm)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().item()

        new_idx = len(pred_labels[pred_labels == predicted])
        pred_labels = np.append(pred_labels, predicted)


        label_folder = os.path.join(stolen_labels_folder, str(predicted))
        os.makedirs(label_folder, exist_ok=True)

        save_image(
            images,
            fp=os.path.join(label_folder, f'{new_idx}.png')
        )

(unique, counts) = np.unique(real_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

(unique, counts) = np.unique(pred_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)