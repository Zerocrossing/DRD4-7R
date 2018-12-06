"""
A single instance of a traveling salesman problem
Includes data about the history of the population
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import debug_print as print

class TSP:

    def __init__(self, graph, population_size, crossover_rate, mutation_rate, num_generations=None):
        self.graph              = graph
        self.population_size    = population_size
        self.string_length      = graph.shape[0]
        self.crossover_rate     = crossover_rate
        self.num_parents        = int(population_size * crossover_rate) if (
                                  int(population_size * crossover_rate) % 2 == 0) \
                                  else int(population_size * crossover_rate) - 1
        self.mutation_rate      = mutation_rate
        self.history            = {}
        self.population         = np.zeros(population_size)
        self.fitness            = np.zeros(population_size)
        self.parent_index       = np.zeros(self.num_parents)
        self.children           = np.zeros(self.num_parents)
        self.children_fitness   = np.zeros(self.num_parents)
        self.mutant_index       = np.array([])
        self.mutant_children_index = np.array([])
        self.best_individual = np.zeros(num_generations)
        # optional variables not used by every solution
        self.num_generations    = num_generations
        self.current_generation = 0

    def add_history(self, string, value):
        """
        history maintains a list of population data over time, such as mean fitness, best fitness, ect.
        """
        if string not in self.history:
            self.history[string] = []
        self.history[string].append(value)

    def plot_history(self, string):
        data = self.history.get(string)
        plt.plot(data)
        plt.title(string)
        plt.show()

    def plot(self, data, arr):
        # plt.scatter(*zip(*data))
        x, y = zip(*data)
        for i in range(0, len(arr) - 1):
            plt.plot(x[arr[i]:arr[i + 1]], y[arr[i]:arr[i + 1]], 'ro-')
        plt.show()

