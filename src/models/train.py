import torch
import numpy as np


def train(model, data, epochs=100, print_every=10):
    """
    Trains a given network model with data for a number of epochs 
    INPUT
        model: torch model to be trained
        data: image set for trainning and validation
        epochs (default=100): number of epochs for each cv
        print_every (default=10): show accuracy and loss every epoch iteration
    OUTPUT
        result: validation best performance
    """
    # create the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    # transfer everything to the best device available
    (data, model, criterion) = transfer_to_device(data, model, criterion)
    (X_train, y_train, X_val, y_val) = data

    for epoch in range(epochs):
        optimizer.zero_grad()
        model.train()

        y_pred = model(x_train)
        y_pred = torch.squeeze(y_pred)

        # training loss
        train_loss = criterion(y_pred, y_train)
        train_loss.backward() # backprop
        optimizer.step() # update the weights

        with torch.no_grad(): # turn off grad
            model.eval() # model in eval mode

            # training accuracy
            train_acc_list = calculate_accuracy(y_train, y_pred)
            train_acc = np.sum(train_acc_list) / len(train_acc_list)

            # predicting validation set
            y_val_pred = model(X_val)
            y_val_pred = torch.squeeze(y_val_pred)

            # validation loss
            val_loss = calculate_criterion(y_val, y_val_pred, criterion)
            
            # validation accuracy
            val_acc_list = calculate_accuracy(y_val, y_val_pred)
            val_acc = np.sum(val_acc_list) / len(val_acc_list)

            if (epoch+1) % print_every == 0:
                tr_lss = round_tensor(train_loss)
                tr_acc = round_tensor(train_acc)
                vl_lss = round_tensor(val_loss)
                vl_acc = round_tensor(val_acc)

                print('EPOCH {}:'.format(epoch+1))
                print('Train Set --- loss:{}; acc:{}'.format(tr_lss, tr_acc))
                print('Validation Set --- loss:{}; acc:{}'.format(vl_lss, vl_acc))