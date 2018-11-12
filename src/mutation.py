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
        elif method_str.lower() == "flip":
            self.method = self.flip
            print("Substring flip method selected for mutation")
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

    def swap(self, population):
        rnd = np.random.randint(0, self.tsp.string_length, size=(population.shape[0], 2))
        for num, vals in enumerate(rnd):
            population[num][vals[0]], population[num][vals[1]] = population[num][vals[1]], population[num][vals[0]]
        return population

    #TODO: numpy me
    def flip(self, population):
        """
        flips the order of a substring of the population between 2 points
        """
        for num, vals in enumerate(population):
            start = np.random.randint(1, self.tsp.string_length - 2)
            end = np.random.randint(2, self.tsp.string_length - start) + start
            flip = np.flip(vals[start:end], axis=0)
            population[num][start:end] = flip
        return population
