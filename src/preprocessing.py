import numpy as np

# crude start to preprocessing
def euclidean(a,b):
    """
    Returns the euclidean distance between two points
    """
    return np.linalg.norm(a-b)

# Need to numpy the hell out of this
def precalculate_distances(data):
    """
    Top Secret Data!
    :param data:
    :return: dictionary of distances between each pair (smaller,larger)
    """

    distances = {}

    for a in range(len(data)-1):
        for b in range(a+1, len(data)):
            key = (a, b)
            distances[key] = euclidean(data[a], data[b])

    return distances

