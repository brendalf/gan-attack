import sys
import click
import torch

from dataset.cifar_data import get_trainset, get_testset
from models.target.cifar10.custom import Cifar10


@click.command()
@click.option(
    '--model',
    default='Cifar10',
    type=click.Choice(['Cifar10'], case_sensitive=False),
    help='The type of model we want to train')
@click.option(
    '--arch',
    default='Custom',
    type=click.Choice(['Custom'], case_sensitive=False),
    help='The type of model architecture we want to train')
def train(model, arch):
    if model == 'Cifar10' and arch == 'Custom':
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
    '--arch',
    default='Custom',
    type=click.Choice(['Custom'], case_sensitive=False),
    help='The type of model architecture we want to evaluate')
@click.option(
    '--path',
    default='models/target/cifar10.custom.pth',
    type=click.Path(),
    help='Path to the trained model')
def evaluate(model, arch, path):
    if model == 'Cifar10' and arch == 'Custom':
        testset = get_testset(batch=32)
        cifar10 = torch.load(path)
        cifar10.evaluate(testset['loader'])



@click.group()
def target():
    pass


target.add_command(evaluate)
target.add_command(train)


if __name__ == "__main__":
    sys.exit(target())