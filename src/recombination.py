"""
Sexual (binary) reproduction of individuals to create new population candidates
"""

import numpy as np
from src.utils import *
from numba import njit, prange


class Recombination:

    def __init__(self, tsp_instance, method_str):
        self.tsp = tsp_instance
        self.set_method(method_str)

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
        else:
            raise Exception("Incorrect method selected for recombination")

    def recombine(self):
        start_timer("recombination")
        selected_parents = self.tsp.parent_index
        parents = self.tsp.population[selected_parents]
        self.tsp.children = self.method(parents)
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


# Parallel adds overhead which can negatively impact performance with low numbers of parents
# for example: with the medium dataset and 100 parents per generation, parallel=True decreases performance
#              however with 200 parents it improves it slightly
@njit(parallel=False, fastmath=True)
def order_crossover(parents):
    children = np.empty_like(parents, dtype=np.uint16)
    for n in prange(parents.shape[0] // 2):
        p1 = parents[n * 2]
        p2 = parents[n * 2 + 1]
        start = np.random.randint(1, p1.size // 2)
        slice = np.arange(start, start + p1.size // 2)
        c1 = order_xover_2par(p1, p2, slice)
        c2 = order_xover_2par(p2, p1, slice)
        children[n * 2] = c1
        children[n * 2 + 1] = c2
    return children


@njit(parallel=False, fastmath=True)
def order_xover_2par(p1, p2, slice):
    inv_slice = in1d(np.arange(p1.size),slice)
    child = np.zeros_like(p1)
    child[slice] = p1[slice]
    p2_diff = in1d(p2,child[slice])
    child[inv_slice] = p2[p2_diff]
    return child


@njit(parallel=False, fastmath=True)
def in1d(array, remove):
    out = np.empty(array.shape[0], dtype=np.bool_)
    remove = set(remove)
    for i in prange(array.shape[0]):
        if array[i] in remove:
            out[i] = False
        else:
            out[i] = True
    return out
