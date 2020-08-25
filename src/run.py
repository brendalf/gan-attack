import sys
import click
import torch

from .adversary.cli import adversary
from .target.cli import target


@click.group()
def main():
    pass


main.add_command(adversary)
main.add_command(target)


if __name__ == "__main__":
    sys.exit(main())