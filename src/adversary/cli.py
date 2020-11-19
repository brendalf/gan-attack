import sys
import click
import torch
import traceback

from models.adversary.copycat import copycat_train, copycat_steal
from models.adversary.knockoff import knockoff_train, knockoff_steal

from models.target.cifar10 import Cifar10Custom

from dataloader.cifar10 import get_testset
from dataloader.imagenet2012 import get_loader

from models.evaluate import evaluate_network

METHODS = {
    'copycat': {
        'steal': copycat_steal,
        'train': copycat_train,
        'stolen_labels': 'data/stolen_labels/copycat.csv'
    },
    'knockoff': {
        'steal': knockoff_steal,
        'train': knockoff_train,
        'stolen_labels': 'data/stolen_labels/knockoff.csv'
    }
}

MODELS = {
    'cifar10': {
        'path': 'models/adversary/',
        'testset': get_testset,
        'archs': {
            'custom': Cifar10Custom
        }
    }
}


@click.command()
@click.option(
    '--method',
    default='copycat',
    type=click.Choice(['copycat', 'knockoff'], case_sensitive=False),
    help='The type of adversary method')
@click.option(
    '--target',
    default='models/target/cifar10.custom.pth',
    type=click.Path(),
    help='Path to the target model')
@click.option(
    '--imagelist',
    default='data/ILSVRC2012_min.csv',
    type=click.Path(),
    help='Path to image list file')
@click.option(
    '--batch',
    default=32,
    type=int,
    help='Batch size for stealing')
def steal(method, target, imagelist, batch):
    method = method.lower()

    try:
        network = torch.load(target)
        loader = get_loader(imagelist, batch=batch)
        METHODS[method]['steal'](network, loader, METHODS[method]['stolen_labels'])
    except:
        traceback.print_exc()
        print("Method {} not defined".format(method))


@click.command()
@click.option(
    '--method',
    default='copycat',
    type=click.Choice(['copycat', 'knockoff'], case_sensitive=False),
    help='The type of adversary method')
@click.option(
    '--model',
    default='cifar10',
    type=click.Choice(['cifar10'], case_sensitive=True),
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
def train(method, model, arch, batch, epochs):
    method = method.lower()
    model = model.lower()
    arch = arch.lower()
    
    try:
        output_path = f"{MODELS[model]['path']}{model}.{arch}.stoled.pth"
        network = MODELS[model]['archs'][arch]()
        loader = get_loader(METHODS[method]['stolen_labels'], batch=batch)
        METHODS[method]['train'](network, loader, output_path, epochs)
    except KeyError:
        print('Model {}.{} not defined'.format(model, arch))
    except:
        traceback.print_exc()


@click.command()
@click.option(
    '--model',
    default='cifar10',
    type=click.Choice(['cifar10'], case_sensitive=True),
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
        saved_path = f"{MODELS[model]['path']}{model}.{arch}.stoled.pth"
        testset = MODELS[model]['testset'](batch=batch)
        network = torch.load(saved_path)
        evaluate_network(network, testset)
    except:
        traceback.print_exc()
        print('Model {} not found. You have to train the model first'.format(saved_path))


@click.group()
def adversary():
    pass


adversary.add_command(steal)
adversary.add_command(evaluate)
adversary.add_command(train)


if __name__ == "__main__":
    sys.exit(adversary())