"""
Saving and loading of files from disk
"""

import numpy as np

def parse_file(filename):
    """
    Parse file reads the relevant TSP .txt files and returns a numpy array representing the graph of cities
    """
    out = []
    with open(filename) as f:
        for line in f:
            nums = [float(num) for num in line.split()[1:3]]
            out.append(nums)
    return np.array(out)

def save_population(filename, population):
    """
    saves the .np files representing a population of candidate solutions
    """
    pass

def load_population(filename):
    """
    loads a population from the same format as saved by save_population
    this allows for offline non-realtime continuation of training, should the need arise
    """