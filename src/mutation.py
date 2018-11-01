"""
Introducing small amounts of randomness to the population
"""

import numpy as np

#TODO: Need to modify to accept fitness

class Mutation:

    def __init__(self, str_length, mutation_rate, method_str, mutation_strength=1.0):
        self.str_length = str_length
        self.rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the mutate function
        """
        if method_str.lower() == "swap":
            self.method = self.swap
            print("Swap method selected for mutation")
        else:
            raise Exception("Incorrect method selected for mutation")

    def mutate(self, population):
        selection = np.random.random_sample(population.shape[0]) < self.rate
        selection = np.argwhere(selection).flatten()
        if selection.size == 0: return population, []
        population[selection] = self.method(population[selection])
        return population, selection

    # todo: there has to be a numpy-esque way to do this
    def swap(self, population):
        rnd = np.random.randint(0, self.str_length, size=(population.shape[0], 2))
        for num, vals in enumerate(rnd):
            population[num][vals[0]], population[num][vals[1]] = population[num][vals[1]], population[num][vals[0]]
        return population
