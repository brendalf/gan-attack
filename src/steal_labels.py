import os
import torch
import numpy as np

from tqdm import tqdm

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, Normalize, ToTensor

from models.vgg import VGG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(suppress=True)

model = VGG('VGG19')
model = torch.load('models/target/cifar10.vgg19.pth')
model = model.to(DEVICE)

norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

dataset_folder = 'data/dataset_gan116'
stolen_labels_folder = 'data/dataset_gan116_sl'

transform = Compose([
    #Resize((32, 32)),
    ToTensor(),
])

dataset = ImageFolder(root=dataset_folder, transform=transform)

if not os.path.exists(stolen_labels_folder):
    os.mkdir(stolen_labels_folder)

real_labels = np.array([])
pred_labels = np.array([])

true_labels = False

print('Generating labels from target...')
with torch.no_grad():
    model.eval()

    for image, label in tqdm(dataset):
        real_labels = np.append(real_labels, label)
        image = image.to(DEVICE)

        # simulando o processamento da api, pois eu n conheco a norm e preciso salvar a imagem original
        images_norm = norm(image.view(3, 32, 32)).view(1, 3, 32, 32)

        outputs = model(images_norm)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().item()

        if (true_labels) and (label != predicted):
            continue

        new_idx = len(pred_labels[pred_labels == predicted])
        pred_labels = np.append(pred_labels, predicted)


        label_folder = os.path.join(stolen_labels_folder, str(predicted))
        os.makedirs(label_folder, exist_ok=True)

        save_image(
            image,
            fp=os.path.join(label_folder, f'{new_idx}.png')
        )

(unique, counts) = np.unique(real_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

(unique, counts) = np.unique(pred_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)
