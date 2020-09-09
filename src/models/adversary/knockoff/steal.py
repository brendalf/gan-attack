import torch

from tqdm import tqdm

def knockoff_steal(model, loader, output_labels):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print('Generating labels from target...')
    with torch.no_grad():
        model.eval()
        with open(output_labels, 'w') as output_fd:
            for images, _, filenames in tqdm(loader):
                images = images.to(device)
                outputs = model(images)
                output_fd.writelines(
                    ['{},{}\n'.format(img_fn, ','.join(labels.cpu().numpy().astype(str))) for img_fn, labels 
                        in zip(filenames, outputs)
                    ]
                )