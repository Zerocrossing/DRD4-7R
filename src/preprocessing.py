import numpy as np
from src.file_utils import parse_file

# crude start to preprocessing

def euclidean(a,b):
    return np.linalg.norm(a-b)

def get_secret_stuff(data):

    distances = {}

    for a in range(len(data)-1):
        for b in range(a+1, len(data)):
            key = (a, b)
            distances[key] = euclidean(data[a], data[b])

    return distances

