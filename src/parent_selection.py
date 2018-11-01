"""
Selecting parents for sexual (non-unary) reproduction
"""

import numpy as np


class Parent_Selection:

    def __init__(self, pop_size, str_len, method_str):
        self.pop_size = pop_size
        self.str_len = str_len
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
            raise Exception("Incorrect method selected for Recombination")

    def select(self, population, num_parents):
        return self.method(population, num_parents)

    def random(self, population, num_parents):
        return np.random.choice(self.pop_size, num_parents, replace=False)
