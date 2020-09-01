import sys
import click
import torch

from models.adversary.copycat.steal import steal as steal_labels


@click.command()
@click.option(
    '--method',
    default='CopyCat',
    type=click.Choice(['CopyCat', 'KnockOff'], case_sensitive=False),
    help='The type of method')
@click.option(
    '--imagelist',
    default='src/dataset/images.txt',
    type=click.Path(),
    help='Path to file with image list')
@click.option(
    '--target',
    default='models/target/cifar10.custom.pth',
    type=click.Path(),
    help='Path to the target model')
def steal(method, imagelist, target):
    if method == 'CopyCat':
        steal_labels(target, imagelist, 'models/adversary/stolen.txt')


@click.command()
def train():
    pass


@click.command()
def evaluate():
    pass


@click.group()
def adversary():
    pass


adversary.add_command(steal)
adversary.add_command(evaluate)
adversary.add_command(train)


if __name__ == "__main__":
    sys.exit(adversary())