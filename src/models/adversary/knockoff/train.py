import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(pred, soft_targets, weights=None):
    return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def knockoff_train(model, dataloader, output_path, epochs=10):
    """
    Trains a given network model with data for a number of epochs and
    save in a default location
    INPUT
        model: the network pytorch model
        dataloader: the train set dataloader
        output_path: path to save the trained model
        epochs (default=10): number of epochs
    """
    criterion = soft_cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('Training model...')
    model = model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm.tqdm(dataloader) as tqdm_loader:
            for i, data in enumerate(tqdm_loader):
                inputs, labels, _ = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 200 == 0:
                    tqdm_loader.set_description('Epoch: {}/{} Loss: {:.3f}'.format(
                        epoch+1, epochs, running_loss))

    print('Finished Training')

    print(f'Saving model to {output_path}')
    torch.save(model, output_path)