"""
Sexual (binary) reproduction of individuals to create new population candidates
"""

import numpy as np
from src.utils import *
from numba import njit, prange
from src.utils import debug_print as print


class Recombination:

    def __init__(self, tsp_instance, method_str, evaluator):
        self.tsp = tsp_instance
        self.set_method(method_str)
        self.dist_cache = evaluator.dist_cache

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "cut_and_crossfill":
            self.method = self.cut_and_crossfill
            print("Cut and crossfill method selected for recombination")
        elif method_str.lower() == "order_crossover":
            self.method = order_crossover
            print("Order crossover method selected for recombination")
        elif method_str.lower() == "scx":
            self.method = scx
            print("Sequential constructive crossover operator selected for recombination")
        elif method_str.lower() == "pmx":
            self.method = pmx
            print("PMX method selected for recombination")
        else:
            raise Exception("Incorrect method selected for recombination")

    def recombine(self):
        start_timer("recombination")
        selected_parents = self.tsp.parent_index
        parents = self.tsp.population[selected_parents]
        self.tsp.children = self.method(parents, self.dist_cache)
        add_timer("recombination")

    def cut_and_crossfill(self, parent1, parent2):
        """
        cut and crossfill method
        Note: ONLY works with permutations, not random strings
        """
        cut = np.random.randint(1, len(parent1) - 1)
        offspring1, offspring2 = np.empty(len(parent1), int), np.empty(len(parent2), int)
        offspring1.fill(-1)
        offspring2.fill(-1)
        offspring1[:cut] = parent1[:cut]
        offspring2[:cut] = parent2[:cut]
        while -1 in offspring1 or -1 in offspring2:
            if parent2[cut] not in offspring1:
                offspring1[np.nonzero(offspring1 == -1)[0][0]] = parent2[cut]  # index of first nonzero
            if parent1[cut] not in offspring2:
                offspring2[np.nonzero(offspring2 == -1)[0][0]] = parent1[cut]
            cut = (cut + 1) % len(parent1)
        return offspring1, offspring2


"""Order Crossover"""
"""###############"""
# Parallel adds overhead which can negatively impact performance with low numbers of parents
# for example: with the medium dataset and 100 parents per generation, parallel=True decreases performance
#              however with 200 parents it improves it slightly

@njit(parallel=True, fastmath=True)
def order_crossover(parents, _):
    children = np.empty_like(parents, dtype=np.uint16)
    for n in prange(parents.shape[0] // 2):
        p1 = parents[n * 2]
        p2 = parents[n * 2 + 1]
        start = np.random.randint(1, p1.size // 2)
        slice = np.zeros_like(p1)
        slice[start:start+p1.size//2] = 1
        slice_bool = ~(slice == 0)
        c1 = order_xover_2par(p1, p2, slice_bool)
        c2 = order_xover_2par(p2, p1, slice_bool)
        children[n * 2] = c1
        children[n * 2 + 1] = c2
    return children

@njit(parallel=True, fastmath=True)
def scx(parents, cache):
    children = np.empty_like(parents, dtype=np.uint16)
    for n in range(parents.shape[0] // 2):
        p1 = parents[n * 2]
        p2 = parents[n * 2 + 1]
        c1 = scx_xover(p1, p2, cache)
        c2 = scx_xover(p2, p1, cache)
        children[n * 2] = c1
        children[n * 2 + 1] = c2
    return children

@njit(parallel=False, fastmath=True)
def scx_xover(p1, p2, cache):
    child = -np.ones_like(p1)
    child[0] = p1[0]
    out_set = set(np.arange(len(child)))
    out_set.remove(child[0])
    in_set = {child[0]}
    # optimization - look up tables instead of using np.where
    p1_lookup, p2_lookup = np.empty_like(p1), np.empty_like(p2)

    index = 0
    for x in p1:
        p1_lookup[x] = p1[(index+1) % len(p1)]
        index += 1

    # optimization - look up tables instead of using np.where
    index = 0
    for x in p2:
        p2_lookup[x] = p2[(index+1) % len(p2)]
        index += 1

    index = 0
    while len(out_set) > 0:
        current_node = child[index]
        node_1 = p1_lookup[current_node]

        if node_1 in in_set:
            node_1 = out_set.pop()
            out_set.add(node_1)

        node_2 = p2_lookup[current_node]
        if node_2 in in_set:
            node_2 = out_set.pop()
            out_set.add(node_2)

        index += 1
        if cache[current_node, node_1] <= cache[current_node, node_2]:
            child[index] = node_1
            in_set.add(node_1)
            out_set.remove(node_1)
        else:
            child[index] = node_2
            in_set.add(node_2)
            out_set.remove(node_2)

    return child



@njit(parallel=False, fastmath=True)
def order_xover_2par(p1, p2, slice):
    """
    Performs order crossover between two parents, whose carry forward
    slice has already been defined

    :param p1:
    :param p2:
    :param slice: A boolean np.array identifying the carry forward slice with `True`
    :return: The child resulting from the order crossover operation
    """
    child = np.empty_like(p1)
    child[slice] = p1[slice]
    p2_diff = ~in1d(p2,child[slice])
    child[~slice] = p2[p2_diff]
    return child


@njit(parallel=False, fastmath=True)
def in1d(array, values):
    out = np.empty(array.shape[0], dtype=np.bool_)
    values = set(values)
    for i in prange(array.shape[0]):
        out[i] = array[i] in values
    return out

"""      PMX      """
"""###############"""
@njit(parallel=True, fastmath=True)
def pmx(parents, _):
    children = np.empty_like(parents, dtype=np.uint16)
    for n in prange(parents.shape[0] // 2):
        p1 = parents[n * 2]
        p2 = parents[n * 2 + 1]
        c1 = pmx_2par(p1, p2)
        c2 = pmx_2par(p2, p1)
        children[n * 2] = c1
        children[n * 2 + 1] = c2
    return children

@njit(parallel=False, fastmath=True)
def pmx_2par(p1, p2):
    """
    Performs PMX between two parents

    :param p1:
    :param p2:
    :return: The child resulting from the PMX crossover operation
    """
    child = -np.ones_like(p1) #max unsigned value

    sliceStart = np.random.randint(1, p1.size // 2)
    sliceEnd = sliceStart + p1.size // 2
    slice = np.zeros_like(p1, np.bool_) #False np array
    slice[sliceStart : sliceEnd] = True

    child[slice] = p1[slice]
    p2SliceDiff = np.zeros_like(p1, np.bool_)
    p2SliceDiff[slice] = ~in1d(p2[slice], child[slice])

    r_index, = np.where(p2SliceDiff)
    for n in prange(r_index.size):
        index = r_index[n]
        targetIndex, = np.where(p2 == p1[index])[0]
        while(slice[targetIndex]): #targetIndex is in the slice
            targetIndex, = np.where(p2 == p1[targetIndex])[0]

        child[targetIndex] = p2[index]

    child[child == -np.uint16(1)] = p2[child == -np.uint16(1)]
    return child

