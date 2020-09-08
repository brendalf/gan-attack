from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils.data import DataLoader

from dataset.imagenet import ImagenetDataset

def get_transform():
    transform = Compose([
        Resize((32,32)),
        ToTensor()        
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