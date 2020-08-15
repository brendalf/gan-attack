import torch
import numpy as np


def transfer_to_device(data, network, criterion):
    """
    Transfer data, network and criteterion to the best device available
    INPUT
        data: torch data
        network: network model
        criterion: loss function
    OUTPUT
        data: torch data transfered
        network: network model transfered
        criterion: loss function transfered
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (X_train, y_train, X_val, y_val) = data

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    network = network.to(device)
    criterion = criterion.to(device)

    trasfered_data = (X_train, y_train, X_val, y_val)
    return (trasfered_data, network, criterion)


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy
    INPUT
        ytrue: tensor array with ground truth
        ypred: tensor array with predicted values
    OUTPUT
        accuracy
    """
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(tensor, decimal_places=3):
    """
    Round tensor to decimal places
    INPUT
        tensor: tensor to be rounded
        decimal_places (default=3): number of decimal places
    OUTPUT
        tensor: rounded tensor
    """
    return round(tensor.item(), decimal_places)