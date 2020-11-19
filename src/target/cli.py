import sys
import click
import torch
import traceback

from target.custom import Custom

from dataloader.cifar10 import get_trainset as cifar10_trainset, get_testset as cifar10_testset
from dataloader.stl10 import get_trainset as stl10_trainset, get_testset as stl10_testset

from models.target.train import train_network
from models.evaluate import evaluate_network


MODELS = {
    'cifar10': {
        'path': 'models/target/',
        'trainset': cifar10_trainset,
        'testset': cifar10_testset,
        'archs': {
            'custom': Custom
        }
    },
    'stl10': {
        'path': 'models/target/',
        'trainset': stl10_trainset,
        'testset': stl10_testset,
        'archs': {
            'custom': Custom
        }
    }
}


@click.command()
@click.option(
    '--model',
    default='cifar10',
    type=click.Choice(['cifar10', 'stl10'], case_sensitive=True),
    help='The type of model we want to train')
@click.option(
    '--arch',
    default='custom',
    type=click.Choice(['custom'], case_sensitive=True),
    help='The type of model architecture we want to train')
@click.option(
    '--batch',
    default=32,
    type=int,
    help='Batch size for training')
@click.option(
    '--epochs',
    default=10,
    type=int,
    help='Number of epochs for training')
def train(model, arch, batch, epochs):
    model = model.lower()
    arch = arch.lower()

    try:
        output_path = f"{MODELS[model]['path']}{model}.{arch}.pth"
        trainset = MODELS[model]['trainset'](batch=batch)
        network = MODELS[model]['archs'][arch]()
        train_network(network, trainset, output_path, epochs)
    except KeyError:
        print('Model {}.{} not defined'.format(model, arch))
    except:
        traceback.print_exc()


@click.command()
@click.option(
    '--model',
    default='cifar10',
    type=click.Choice(['cifar10', 'stl10'], case_sensitive=True),
    help='The type of model we want to evaluate')
@click.option(
    '--arch',
    default='custom',
    type=click.Choice(['custom'], case_sensitive=True),
    help='The type of model architecture to evaluate')
@click.option(
    '--batch',
    default=32,
    type=int,
    help='Batch size for testing')
def evaluate(model, arch, batch):
    model = model.lower()
    arch = arch.lower()

    try:
        saved_path = f"{MODELS[model]['path']}{model}.{arch}.pth"
        testset = MODELS[model]['testset'](batch=batch)
        network = torch.load(saved_path)
        evaluate_network(network, testset)
    except:
        traceback.print_exc()
        print('Model {} not found. You have to train the model first'.format(saved_path))



@click.group()
def target():
    pass


target.add_command(evaluate)
target.add_command(train)


if __name__ == "__main__":
    sys.exit(target())