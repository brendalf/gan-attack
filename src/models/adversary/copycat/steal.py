import torch

from tqdm import tqdm

def copycat_steal(model, loader, output_labels):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print('Generating labels from target...')
    with torch.no_grad():
        model.eval()
        with open(output_labels, 'w') as output_fd:
            for images, _, filenames in tqdm(loader):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                output_fd.writelines(
                    ['{},{}\n'.format(img_fn, label) for img_fn, label 
                        in zip(filenames, predicted)
                    ]
                )