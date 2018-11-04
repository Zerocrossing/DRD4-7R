"""
Selecting a specific number of members from a population to continue to the next generation
"""
import numpy as np
from src.utils import *


class Termination:

    def __init__(self, num_generation, time_limit, method_str):
        self.max_generation = num_generation
        self.max_time = time_limit
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the select function
        :return:
        """
        if method_str.lower() == "num_iterations":
            self.method = self.num_iterations
            print("Fixed number of iterations method selected for termination")
        elif method_str.lower() == "time_limit":
            self.method = self.time_limit
            print("Fixed time limit method selected for termination")
        else:
            raise Exception("Incorrect method selected for termination")

    def select(self, current_generation, current_time):
        start_timer("termination")
        hasta_la_vista = self.method(current_generation, current_time)
        add_timer("termination")
        return hasta_la_vista

    def num_iterations(self, current_generation, current_time):
        return current_generation < self.max_generation

    def time_limit(self, current_generation, current_time):
        return current_time < self.max_time