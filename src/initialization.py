"""
Instantiation of candidate solutions
Current methods include:
    - random
"""

import numpy as np
from src.utils import *
from src.utils import debug_print as print, set_debug
from numba import njit, prange
from src.preprocessing import precalculate_distances
from src.utils import debug_print as print


class Initialization:

    def __init__(self, tsp_instance, method_str):
        self.tsp = tsp_instance
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "random_permutations":
            self.method = self.random_permutations
            print("Random permutations method selected for initialization")
        elif method_str.lower() == "demo_random":
            self.method = self.DEMO_random
        elif method_str.lower() == "greedy_neighbour":
            self.method = self.greedy_neighbour
            print("Greedy Neighbour method selected for initialization")
        else:
            raise Exception("Incorrect method selected for initialization")

    def initialize(self):
        """
        Calls whichever method is currently selected
        """
        start_timer("initialization")
        self.tsp.population = self.method()
        add_timer("initialization")

    ##### INITIALIZATION METHODS #####

    def DEMO_random(self):
        """
        randomized candidate solutions
        :return:
        """
        return np.random.randint(1, 50, (self.tsp.population_size, self.tsp.string_length))

    def random_permutations(self):
        """
        randomized permutation candidate solutions
        """
        return np.array(
            [np.random.permutation(np.arange(0, self.tsp.string_length)) for n in np.arange(self.tsp.population_size)],
            dtype=np.uint16)

    def greedy_neighbour(self):
        distances = precalculate_distances(self.tsp.graph)
        population = self.greedy_jit(self.tsp.population_size, distances, self.tsp.string_length)
        return population

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def greedy_jit(pop_size, distances, str_len):
        population = np.empty((pop_size, str_len), dtype=np.uint16)
        for n in prange(pop_size):
            init = np.random.randint(0, str_len)
            population[n] = np.argsort(distances[init])
        return population