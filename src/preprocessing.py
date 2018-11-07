import numpy as np

def precalculate_distances(data):
    """
    returns a numpy array where [x][y] is the distance between those two indices in the dataset
    for example array[1][5] will return the distance between the first and fifth cities
    """
    dim = data.shape[0]
    out = np.zeros((dim,dim))
    for n in np.arange(dim):
        dist = np.linalg.norm(data[n]-data,axis=1)
        out[n]= dist
    return out

