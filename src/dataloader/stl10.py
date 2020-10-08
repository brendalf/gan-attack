from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

def get_transform():
    transform = Compose([
        Resize((32,32)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
    ])
    
    return transform


def get_trainset(root='./data', batch=4):
    transform = get_transform()

    trainset = STL10(
        root=root, split='train', download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2
    )
    
    return trainloader


def get_testset(root='./data', batch=4):
    transform = get_transform()
    
    testset = STL10(
        root=root, split='test', download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=2
    )
    
    return testloader