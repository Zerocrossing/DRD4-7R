"""
Evaluation heuristics
"""
import numpy as np
from src.utils import *
from src.preprocessing import precalculate_distances
from numba import jit, prange
from src.utils import debug_print as print

class Evaluation:

    def __init__(self, tsp_instance, method_str):
        self.tsp = tsp_instance
        self.set_method(method_str)
        self.dist_cache = precalculate_distances(tsp_instance.graph)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "cached_euclidean":
            self.method = dist_from_cache
            print("Cached euclidean distance method selected for evaluation")
        else:
            raise Exception("Incorrect method selected for evaluation")

    def evaluate(self, use_mask=False):
        start_timer("evaluation")
        if not use_mask:
            self.tsp.fitness = self.method(self.tsp.population, self.dist_cache)
        # else we only evaluate individuals who are indicated to be evaluated
        elif self.tsp.mutant_index.size != 0:
            self.tsp.fitness[self.tsp.mutant_index] = self.method(self.tsp.population[self.tsp.mutant_index], self.dist_cache)
        add_timer("evaluation")

    def evaluate_children(self, use_mask=False):
        start_timer("evaluation")
        if not use_mask:
            self.tsp.children_fitness = self.method(self.tsp.children, self.dist_cache)
        elif self.tsp.mutant_children_index.size != 0:
            self.tsp.children_fitness[self.tsp.mutant_children_index] = self.method(
                self.tsp.children[self.tsp.mutant_children_index], self.dist_cache)
        add_timer("evaluation")


    def vertex_distance(self, point1, point2):
        return self.dist_cache[point1, point2]

@jit(nopython=True, parallel=True, fastmath=True)
def dist_from_cache(population, cache):
    out = np.zeros(population.shape[0],dtype=np.float64)
    for n in prange(population.shape[0]):
        p1 = population[n]
        out[n] = single_dist(p1,cache)
    return out

@jit(nopython=True, parallel=False, fastmath=True)
def single_dist(path,cache):
    out = 0
    for n in prange(path.size):
        out -= cache[path[n], path[(n + 1) % path.size]]
    return out
