"""
Selecting a specific number of members from a population to continue to the next generation
"""
import numpy as np
from src.utils import *
from src.utils import debug_print as print

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
        elif method_str.lower() == "mu_comma_lambda":
            self.method = self.mu_comma_lambda
            print("mu,lambda method selected for survivor selection")
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
        offspringSize = self.tsp.children_fitness.shape[0]
        combo_pop = np.append(self.tsp.population, self.tsp.children, axis=0)
        combo_fitness = np.append(self.tsp.fitness, self.tsp.children_fitness)

        # Get the best mu individuals (i.e. exclude the worst lambda individuals)
        survivorIndices = np.argpartition(combo_fitness, kth=offspringSize)[offspringSize:]
        self.tsp.population = combo_pop[survivorIndices]
        self.tsp.fitness = combo_fitness[survivorIndices]

    def mu_comma_lambda(self):
        offspringSize = self.tsp.children_fitness.shape[0]

        #Get all individuals from the current population except for the lowest `offspringSize` individuals
        survivorIndices = np.argpartition(self.tsp.fitness, kth=offspringSize)[offspringSize:]
        survivors, survivorFitness = self.tsp.population[survivorIndices], self.tsp.fitness[survivorIndices]

        #Establish next generation
        self.tsp.population = np.append(survivors, self.tsp.children, axis=0)
        self.tsp.fitness = np.append(survivorFitness, self.tsp.children_fitness)
