import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score

def evaluate_network(model, dataloader, total):
    correct = 0
    res_pos = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    results = np.zeros([total, 2], dtype=np.int)

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            results[res_pos:res_pos+labels.size(0), :] = np.array([labels.tolist(), predicted.tolist()]).T
            res_pos += labels.size(0)

            correct += (predicted == labels).sum().item()

    micro_avg = f1_score(results[:,0], results[:,1], average='micro')
    macro_avg = f1_score(results[:,0], results[:,1], average='macro')
    print('Average: {:.2f}% ({:d} images)'.format(100. * (correct/total), total))
    print('Micro Average: {:.6f}'.format(micro_avg))
    print('Macro Average: {:.6f}\n'.format(macro_avg))

def evaluate_network_with_classes(model, dataloader, total):
    def micro_avg_class(x):
        results_class = results[np.where(results[:, 0] == x)]
        aux = np.where(results_class[:, 1] == x)
        results_class[:, 1] = 0
        results_class[aux, 1] = x
        return f"{(100*f1_score(results_class[:,0], results_class[:,1], average='micro')):.2f}%"
    def acc_by_class(x):
        aux = results[np.where(results[:, 0] == x)]
        x_true = len(aux)
        x_pred = len(aux[np.where(aux[:, 1] == x)])
        return f"{(100*x_pred/x_true):.2f}%"

    correct = 0
    res_pos = 0

    #best device available
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    results = np.zeros([total, 2], dtype=np.int)

    print('Evaluating model...')
    model = model.to(device)
    with torch.no_grad(): # turn off grad
        model.eval() # network in evaluation mode

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            results[res_pos:res_pos+labels.size(0), :] = np.array([labels.tolist(), predicted.tolist()]).T
            res_pos += labels.size(0)

            correct += (predicted == labels).sum().item()

    micro_avg = f1_score(results[:,0], results[:,1], average='micro')
    macro_avg = f1_score(results[:,0], results[:,1], average='macro')
    print('Average: {:.2f}% ({:d} images)'.format(100. * (correct/total), total))
    print(
        'Micro Average: {:.2f}% ['.format(100 * micro_avg),
        ", ".join(map(acc_by_class, np.arange(10))), "]", sep=""
    )
    print('Macro Average: {:.2f}%'.format(100 * macro_avg))