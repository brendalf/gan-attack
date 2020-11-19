import tqdm
import torch

def evaluate_network(model, dataloader):
    """
    Evaluate the trained model with the testset inside dataloader
    INPUT
        model: the trained network pytorch model
        dataloader: the test set dataloader
    OUTPUT
        accuracy: network accuracy
    """
    correct = 0
    total = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm.tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))
    return (100 * correct / total)