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
        self.data = precalculate_distances(tsp_instance.graph)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "precalculate_distances":
            self.method = self.use_preprocessed_array
            print("Euclidean distance method selected for evaluation")
        else:
            raise Exception("Incorrect method selected for evaluation")

    # todo incorporate mutation masking to save time (only calculate fitness for individuals we mutated)
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

    def euclidean(self, a, b):
        """
        euclidean distance
        """
        if a > b:
            a, b = b, a
        return self.data[a, b]

    # Need to use an efficient implementation for this
    def use_preprocessed_array(self, population):
        distance = []
        for individual in population:
            current_distance = 0
            for a in range(len(individual) - 1):
                current_distance -= self.euclidean(individual[a], individual[a + 1])
            distance.append(current_distance)
        return np.array(distance)
