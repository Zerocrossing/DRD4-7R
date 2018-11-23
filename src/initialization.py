"""
Instantiation of candidate solutions
Current methods include:
    - random
"""

import numpy as np
from src.utils import *
from src.utils import debug_print as print, set_debug


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
            print("DEMO_Random method selected for initialization")
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
        return np.array([np.random.permutation(np.arange(0, self.tsp.string_length)) for n in np.arange(self.tsp.population_size)],dtype=np.uint16)
