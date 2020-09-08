def evaluate(self, dataloader):
    """
    Used the trained model to predict new data
    INPUT
        dataloader: torch dataloader
    """
    correct = 0
    total = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('Evaluating model...')
    model = self.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for data in tqdm.tqdm(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))