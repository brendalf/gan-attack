
import os
import torch
import numpy as np

from tqdm import tqdm

import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, Normalize, ToTensor

from models.vgg import VGG16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/target/mnist.vgg16.pth'

np.set_printoptions(suppress=True)

checkpoint = torch.load(MODEL_PATH)
if isinstance(checkpoint, dict):
    model = VGG16(out_features=10)
    model = model.to(DEVICE)
    checkpoint = torch.load(MODEL_PATH)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(checkpoint['net'])
    print("Epoch: ", checkpoint['epoch'] + 1)
else:
    model = torch.load(MODEL_PATH)
    model.to(DEVICE)

SIZE = 32
resize = Resize((SIZE, SIZE))
# norm = Normalize((0.5063, 0.5063, 0.5063), (0.2412, 0.2412, 0.2412))
# norm = Normalize((0.43767047, 0.44375867, 0.47279018), (0.19798356, 0.20096427, 0.19697163))
norm = Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
to_tensor = ToTensor()

true_labels = False
data_folder = 'dataset_gan183'
dataset_folder = f'data/mnist.vgg16/{data_folder}'
stolen_labels_folder = f'data/mnist.vgg16/{data_folder}_sl{"_ml" if true_labels else ""}'

dataset = ImageFolder(root=dataset_folder)

if not os.path.exists(stolen_labels_folder):
    os.mkdir(stolen_labels_folder)

real_labels = np.array([])
pred_labels = np.array([])

print('Generating labels from target...')
with torch.no_grad():
    model.eval()

    for image, label in tqdm(dataset):
        real_labels = np.append(real_labels, label)
        # image = image.to(DEVICE)
        image = resize(image)

        # simulando o processamento da api, pois eu n conheco a norm e preciso salvar a imagem original
        images_resized = to_tensor(image).to(DEVICE)
        images_norm = norm(images_resized.view(3, SIZE, SIZE)).view(1, 3, SIZE, SIZE)
        # images_norm = images_resized.view(1, 3, SIZE, SIZE)

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
            to_tensor(image),
            fp=os.path.join(label_folder, f'{new_idx}.png')
        )

(unique, counts) = np.unique(real_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

(unique, counts) = np.unique(pred_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)
