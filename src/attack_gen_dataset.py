import os
import torch
import numpy as np
from torchvision.datasets.mnist import EMNIST, MNIST

# from tqdm import tqdm
# from shutil import copyfile

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Grayscale

from torchvision.utils import save_image
from torchvision.transforms import Resize, Compose, ToTensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(suppress=True)

new_folder = 'data/mnist_test'

# trainset = ImageFolder(
    # root="./data/FER7/PD_OL",
    # transform=Compose([
        # Resize((32, 32)),
        # ToTensor(),
    # ])
# )
trainset = MNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=Compose([
        # Resize((32, 32)),
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])
)

trainloader = DataLoader(trainset, shuffle=True, batch_size=1)
# testloader = DataLoader(testset, shuffle=True, batch_size=1)

os.makedirs(new_folder, exist_ok=True)

num = 1000

# map_classes = {
    # 0: 0, # airplane
    # 2: 1, # auto
    # 1: 2, # bird
    # 3: 3, # cat
    # 4: 4, # deer
    # 5: 5, # dog
    # # 7: 6, # monkey | frog
    # 6: 7, # horse
    # 8: 8, # sheep
    # 9: 9, # truck
# }

labels = {l:0 for l in np.arange(0, 10)}

print('Generating attack dataset...')
for image, label in trainset:
    # if label in map_classes.keys():
    # label = map_classes[label]

    label_folder = os.path.join(new_folder, str(label))

    if not os.path.exists(label_folder):
        os.mkdir(label_folder)

    if labels[label] == num:
        continue

    save_image(image, fp=os.path.join(label_folder, f'{labels[label]}.png'))
    labels[label] += 1

# for image, label in testset:
    # if label in map_classes.keys():
        # label = map_classes[label]

        # label_folder = os.path.join(new_folder, str(label))

        # if not os.path.exists(label_folder):
            # os.mkdir(label_folder)

        # if labels[label] == num:
            # continue

        # save_image(image, fp=os.path.join(label_folder, f'{labels[label]}.png'))
        # labels[label] += 1

# id_frog = 'n01644373'
# frogs = os.listdir(os.path.join('data/imagenet/', id_frog))
# print(len(frogs))

# os.makedirs(f"{new_folder}/6", exist_ok=True)

# i = 0
# for img_frog in np.random.choice(frogs, num, replace=False):
    # copyfile(f"data/imagenet/{id_frog}/{img_frog}", f"{new_folder}/6/{i}.jpg")
    # i += 1
