"""
Introducing small amounts of randomness to the population
"""

import numpy as np
from src.utils import *
from numba import njit, prange

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
        elif method_str.lower() == "scramble":
            self.method = self.scramble
            print("Scramble method selected for mutation")
        else:
            raise Exception("Incorrect method selected for mutation")

    def mutate_population(self):
        population = self.tsp.population
        bestIndividual = np.argmax(self.tsp.fitness)

        start_timer("mutation")
        selection = np.random.random_sample(population.shape[0]) < self.tsp.mutation_rate
        selection[bestIndividual] = False #Elitism: do not mutate best individual
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
    @staticmethod
    @njit(parallel=False, fastmath=True)
    def flip(population):
        """
        flips the order of a substring of the population between 2 points
        """
        for num in prange(population.shape[0]):
            vals = population[num]
            str_len = population[0].size
            start = np.random.randint(1, str_len - 2)
            end = np.random.randint(2, str_len - start) + start
            flip = vals[end-1:start-1:-1]
            population[num][start:end] = flip
        return population

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def scramble(population):
        for num in prange(population.shape[0]):
            memb = population[num]
            str_len = memb.size
            rand = np.random.randint(0,str_len,2)
            if rand[0]>rand[1]:
                rand = rand[::-1]
            np.random.shuffle(memb[rand[0]:rand[1]])
        return population

