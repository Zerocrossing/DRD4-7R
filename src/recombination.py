"""
Sexual (binary) reproduction of individuals to create new population candidates
"""

import numpy as np
from src.utils import *


class Recombination:

    def __init__(self, pop_size, str_len, method_str):
        self.pop_size = pop_size
        self.str_len = str_len
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "cut_and_crossfill":
            self.method = self.cut_and_crossfill
            print("Cut and crossfill method selected for recombination")
        else:
            raise Exception("Incorrect method selected for recombination")

    def recombine(self, population, selected_parents):
        start_timer("recombination")
        num_parents = len(selected_parents)
        children = np.zeros((num_parents, self.str_len), dtype=np.int)
        for n in np.arange(0, len(selected_parents), 2):
            p1 = population[selected_parents[n]]
            p2 = population[selected_parents[n + 1]]
            c1, c2 = self.method(p1, p2)
            children[n] = c1
            children[n + 1] = c2
        add_timer("recombination")
        return children

    def cut_and_crossfill(self, parent1, parent2):
        """
        cut and crossfill method
        Note: ONLY works with permutations, not random strings
        """
        cut = np.random.randint(1, len(parent1) - 1)
        offspring1, offspring2 = np.empty(len(parent1), int), np.empty(len(parent2), int)
        offspring1.fill(-1)
        offspring2.fill(-1)
        offspring1[:cut] = parent1[:cut]
        offspring2[:cut] = parent2[:cut]
        while -1 in offspring1 or -1 in offspring2:
            if parent2[cut] not in offspring1:
                offspring1[np.nonzero(offspring1 == -1)[0][0]] = parent2[cut]  # index of first nonzero
            if parent1[cut] not in offspring2:
                offspring2[np.nonzero(offspring2 == -1)[0][0]] = parent1[cut]
            cut = (cut + 1) % len(parent1)
        return offspring1, offspring2
