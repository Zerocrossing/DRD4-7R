"""
Selecting parents for sexual (non-unary) reproduction
"""

import numpy as np
from src.utils import *
from src.utils import debug_print as print


class Parent_Selection:

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
            print("Random method selected for parent selection")
        elif method_str.lower() == "mu_plus_lambda":
            self.method = self.mu_plus_lambda
            print("mu+lambda method selected for parent selection")
        elif method_str.lower() == "roulette_wheel":
            self.method = self.roulette_wheel
            print("Roulette Wheel method selected for parent selection")
        else:
            raise Exception("Incorrect method selected for parent selection")

    def select(self):
        start_timer("parent selection")
        self.method()
        add_timer("parent selection")

    def random(self):
        self.tsp.parent_index = np.random.choice(self.tsp.population_size, self.tsp.num_parents, replace=False)

    def roulette_wheel(self):
        popSize = self.tsp.population_size
        windowedFitness = self.tsp.fitness - self.tsp.fitness.min()
        fitnessSum = windowedFitness.sum()
        #The selection probability is fitness proportional if the fitnessSum is not 0. Otherwise, all individuals are the
        #same and we use an equiprobable selection strategy
        selectionProbability = windowedFitness/fitnessSum if fitnessSum != 0 else np.ones_like(windowedFitness)/popSize

        bestIndividual = np.argmax(windowedFitness) #Elitism
        selectedParents = np.random.choice(popSize, size=self.tsp.num_parents-1, replace=True, p=selectionProbability)
        self.tsp.parent_index = np.append(bestIndividual, selectedParents)


    def mu_plus_lambda(self):
        parentIndices = np.argpartition(-self.tsp.fitness, kth=self.tsp.num_parents)[:self.tsp.num_parents]
        self.tsp.parent_index = parentIndices
