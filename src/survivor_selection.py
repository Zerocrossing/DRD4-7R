"""
Selecting a specific number of members from a population to continue to the next generation
"""
import numpy as np


class Survivor_Selection:

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
            print("Random method selected for survivor selection")
        elif method_str.lower() == "mu_plus_lambda":
            self.method = self.mu_plus_lambda
            print("mu+lambda method selected for survivor selection")
        else:
            raise Exception("Incorrect method selected for survivor selection")

    def select(self, population, population_fitness, children, children_fitness):
        return self.method(population, population_fitness, children, children_fitness)

    def random(self, population, population_fitness, children, children_fitness):
        combo_pop = np.append(population, children, axis=0)
        combo_fitness = np.append(population_fitness, children_fitness)
        selection = np.random.randint(population.shape[0], size=population.shape[0])
        return combo_pop[selection], combo_fitness[selection]

    def mu_plus_lambda(self, population, population_fitness, children, children_fitness):
        combo_pop = np.append(population, children, axis=0)
        combo_fitness = np.append(population_fitness, children_fitness)
        args = np.argsort(combo_fitness)
        return combo_pop[args][-population.shape[0]:], combo_fitness[args][-population.shape[0]:]
