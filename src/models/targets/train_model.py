import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import random

import torch.optim as optim

from model import CNN
from cifar_data import get_datasets



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_samples(data_loader, n_batches=1, classes=None):
    # get some random training images
    count = 0
    for images, labels in data_loader:
        count += 1
        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(
                    '{:5s}'.format(
                      str(labels[j].item()) if classes is not None else classes[labels[j]]
                    ) for j in range(4))
            )
        if count == n_batches: break
    
    

