"""
Instantiation of candidate solutions
Current methods include:
    - random
"""

import numpy as np
from src.utils import debug_print as print, set_debug


class Initialization:

    def __init__(self, pop_size, str_len, method_str):
        self.pop_size = pop_size
        self.str_len = str_len
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "random":
            self.method = self.random
            print("Random method selected for Initialization")
        elif method_str.lower() == "demo_random":
            self.method = self.DEMO_random
            print("DEMO_Random method selected for Initialization")
        else:
            raise Exception("Incorrect method selected for Initialization")

    def initialize(self):
        """
        Calls whichever method is currently selected
        """
        population = self.method(self.str_len, self.pop_size)
        return population

    ##### INITIALIZATION METHODS #####

    def DEMO_random(self, str_len=None, pop_size=None):
        """
        randomized candidate solutions
        :return:
        """
        if str_len is None: str_len = self.str_len
        if pop_size is None: pop_size = self.pop_size
        return np.random.randint(1, 50, (pop_size, str_len))

    def random(self, str_len=None, pop_size=None):
        """
        randomized candidate solutions
        :return:
        """
        pass
