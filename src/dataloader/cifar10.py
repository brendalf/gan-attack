from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.datasets import CIFAR10
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

    trainset = CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2
    )
    
    return trainloader


def get_testset(root='./data', batch=4):
    transform = get_transform()
    
    testset = CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=2
    )
    
    return testloader