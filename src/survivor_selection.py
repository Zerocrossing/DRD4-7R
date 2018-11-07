"""
Selecting a specific number of members from a population to continue to the next generation
"""
import numpy as np
from src.utils import *


class Survivor_Selection:

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
            print("Random method selected for survivor selection")
        elif method_str.lower() == "mu_plus_lambda":
            self.method = self.mu_plus_lambda
            print("mu+lambda method selected for survivor selection")
        else:
            raise Exception("Incorrect method selected for survivor selection")

    def select(self):
        start_timer("survivor selection")
        self.method()
        add_timer("survivor selection")

    def random(self):
        combo_pop = np.append(self.tsp.population, self.tsp.children, axis=0)
        combo_fitness = np.append(self.tsp.fitness, self.tsp.children_fitness)
        selection = np.random.randint(self.tsp.population.shape[0], size=self.tsp.population.shape[0])
        self.tsp.population = combo_pop[selection]
        self.tsp.fitness = combo_fitness[selection]

    def mu_plus_lambda(self):
        combo_pop = np.append(self.tsp.population, self.tsp.children, axis=0)
        combo_fitness = np.append(self.tsp.fitness, self.tsp.children_fitness)
        args = np.argsort(combo_fitness)[::-1]
        self.tsp.population = combo_pop[args][:self.tsp.population_size]
        self.tsp.fitness = combo_fitness[args][:self.tsp.population_size]
