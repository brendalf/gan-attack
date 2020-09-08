import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from torchvision.transforms import ToPILImage

class ImagenetDataset(Dataset):
    """ImageNet 2012 Dataset"""

    def __init__(self, csv_file, transform=None):
        """
        INPUT
            csv_file (string): Path to the csv file with annotations
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv_file.iloc[idx, 0]
        image = np.array(cv2.imread(img_name, cv2.IMREAD_COLOR), dtype=np.uint8)
        image = ToPILImage()(image)
        classe = 0

        if self.csv_file.shape[1] > 1:
            classe = self.csv_file.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, classe, img_name