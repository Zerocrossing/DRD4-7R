"""
Evaluation heuristics
"""
import numpy as np
from src.utils import *
from src.preprocessing import precalculate_distances


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
            self.method = self.use_preprocessed_array
            print("Cached euclidean distance method selected for evaluation")
        else:
            raise Exception("Incorrect method selected for evaluation")

    def evaluate(self, use_mask=False):
        start_timer("evaluation")
        if not use_mask:
            self.tsp.fitness = self.method(self.tsp.population)
        # else we only evaluate individuals who are indicated to be evaluated
        elif self.tsp.mutant_index.size != 0:
            self.tsp.fitness[self.tsp.mutant_index] = self.method(self.tsp.population[self.tsp.mutant_index])
        add_timer("evaluation")

    def evaluate_children(self, use_mask=False):
        start_timer("evaluation")
        if not use_mask:
            self.tsp.children_fitness = self.method(self.tsp.children)
        elif self.tsp.mutant_children_index.size != 0:
            self.tsp.children_fitness[self.tsp.mutant_children_index] = self.method(
                self.tsp.children[self.tsp.mutant_children_index])
        add_timer("evaluation")

    def use_preprocessed_array(self, population):
        pop_roll = np.roll(population, 1, axis=1)
        return -self.dist_cache[population, pop_roll].sum(axis=1)
