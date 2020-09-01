import torch
import torchvision.transforms as transforms

from tqdm import tqdm
from sys import argv, exit, stderr

from .image_list import ImageList

def steal(model_path, dataset_path, stolen_labels_path, batch_size=32):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model = model.to(device)
    
    print('Handling images...')
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = ImageList(dataset_path, color=True, transform=transform, return_filename=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    print('Generating labels from target...')
    with torch.no_grad():
        model.eval()
        with open(stolen_labels_path, 'w') as output_fd:
            for images, _, filenames in tqdm(loader):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                output_fd.writelines(['{} {}\n'.format(img_fn, label) for img_fn, label in zip(filenames, predicted)])