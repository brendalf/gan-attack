from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils.data import DataLoader

from dataset.imagenet import ImagenetDataset

def get_transform():
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))          
    ])
    
    return transform


def get_loader(csv_file, batch=4):
    transform = get_transform()

    trainset = ImagenetDataset(
        csv_file=csv_file, transform=transform
    )
    loader = DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2
    )
    
    return loader