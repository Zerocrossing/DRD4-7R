"""
Selecting parents for sexual (non-unary) reproduction
"""

import numpy as np
from src.utils import *


class Parent_Selection:

    def __init__(self, tsp_instance, method_str):
        self.tsp = tsp_instance
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the select function
        :return:
        """
        if method_str.lower() == "random":
            self.method = self.random
            print("Random method selected for parent selection")
        else:
            raise Exception("Incorrect method selected for parent selection")

    def select(self):
        start_timer("parent selection")
        self.tsp.parent_index = self.method()
        add_timer("parent selection")

    def random(self):
        return np.random.choice(self.tsp.population_size, self.tsp.num_parents, replace=False)
