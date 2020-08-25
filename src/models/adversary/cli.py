import sys
import click
import torch

from dataset.cifar_data import get_trainset, get_testset


@click.command()
def train():
    pass


@click.command()
def evaluate():
    pass


@click.group()
def adversary():
    pass


adversary.add_command(evaluate)
adversary.add_command(train)


if __name__ == "__main__":
    sys.exit(adversary())