import sys
import click
import torch

from src.data.cifar_data import get_trainset, get_testset
from .cifar10.custom import Cifar10


@click.command()
@click.option(
    '--model',
    default='Cifar10',
    type=click.Choice(['Cifar10'], case_sensitive=False),
    help='The type of model we want to evaluate')
@click.option(
    '--out',
    default='models/target/cifar_model.pth',
    type=click.Path(),
    help='Path to save trained model')
def train(model, out):
    if model == 'Cifar10':
        trainset = get_trainset(batch=32)
        cifar10 = Cifar10()
        cifar10.fit(trainset['loader'])


@click.command()
@click.option(
    '--model',
    default='Cifar10',
    type=click.Choice(['Cifar10'], case_sensitive=False),
    help='The type of model we want to evaluate')
@click.option(
    '--path',
    default='models/target/cifar_model.pth',
    type=click.Path(),
    help='Path to trained model')
def evaluate(model, path):
    if model == 'Cifar10':
        testset = get_testset(batch=32)
        cifar10 = Cifar10()
        cifar10.load_state_dict(torch.load(path))
        cifar10.evaluate(testset['loader'])



@click.group()
def target():
    pass


target.add_command(evaluate)
target.add_command(train)


if __name__ == "__main__":
    sys.exit(target())