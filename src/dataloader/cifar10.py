from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_trainset(root='./data', batch=32):
    trainset = CIFAR10(
        root=root, 
        train=True, 
        download=True, 
        transform=Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )
    trainloader = DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=2
    )
    
    return trainloader


def get_testset(root='./data', batch=4):
    testset = CIFAR10(
        root=root, 
        train=False, 
        download=True, 
        transform=Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     
        ])
    )
    testloader = DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=2
    )
    
    return testloader