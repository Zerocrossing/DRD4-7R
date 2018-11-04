"""
Introducing small amounts of randomness to the population
"""

import numpy as np
from src.utils import *

#TODO: Need to modify to accept fitness

class Mutation:

    def __init__(self, tsp_instance, method_str, mutation_strength=1.0):
        self.tsp = tsp_instance
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

    def mutate_population(self):
        population = self.tsp.population
        start_timer("mutation")
        selection = np.random.random_sample(population.shape[0]) < self.tsp.mutation_rate
        selection = np.argwhere(selection).flatten()
        if selection.size == 0:
            self.tsp.mutant_index = np.array([])
        else:
            population[selection] = self.method(population[selection])
            self.tsp.mutant_index = selection
        add_timer("mutation")

    def mutate_children(self):
        population = self.tsp.children
        start_timer("mutation")
        selection = np.random.random_sample(population.shape[0]) < self.tsp.mutation_rate
        selection = np.argwhere(selection).flatten()
        if selection.size == 0:
            self.tsp.mutant_index = np.array([])
        else:
            population[selection] = self.method(population[selection])
            self.tsp.mutant_children_index = selection
        add_timer("mutation")

    # todo: there has to be a numpy-esque way to do this
    def swap(self, population):
        rnd = np.random.randint(0, self.tsp.string_length, size=(population.shape[0], 2))
        for num, vals in enumerate(rnd):
            population[num][vals[0]], population[num][vals[1]] = population[num][vals[1]], population[num][vals[0]]
        return population
