import os
import sys

def generate_experiment_id():
    f = open('logs/experiment_id')
    next_id = int(f.readline()) + 1
    f.close()

    f = open('logs/experiment_id', 'w')
    f.write(str(next_id))
    f.close()

    return next_id

class ExperimentLog():
    def __init__(self, experiment_path):
        self.path = experiment_path

    def write(self, log, verbose=True):
        with open(self.path, mode='a') as f:
            f.write(f"{log}\n")

        if verbose:
            print(log)

