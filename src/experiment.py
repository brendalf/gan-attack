import os
import sys

def generate_experiment_id():
    f = open('logs/experiment_id', 'r')
    next_id = int(f.readline()) + 1
    f.close()

    f = open('logs/experiment_id', 'w')
    f.write(next_id)
    f.close()

    return next_id
