import tqdm
import torch
import torch.nn as nn

def train_network(model, dataloader, output_path, epochs=10):
    """
    Trains a given network model with data for a number of epochs and
    save in a default location
    INPUT
        model: the network pytorch model
        dataloader: the train set dataloader
        output_path: path to save the trained model
        epochs (default=10): number of epochs
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('Training model...')
    model = model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm.tqdm(dataloader) as tqdm_train:
            for i, data in enumerate(tqdm_train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 0:
                    tqdm_train.set_description('Epoch: {}/{} Loss: {:.3f}'.format(
                        epoch+1, epochs, running_loss))

    print('Finished Training')

    print(f'Saving model to {output_path}')
    torch.save(model, output_path)