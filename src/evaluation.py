"""
Evaluation heuristics
"""
import numpy as np
from src.utils import *
from src.preprocessing import get_secret_stuff


class Evaluation:

    def __init__(self, graph, method_str, data):
        self.graph = graph
        self.set_method(method_str)
        self.data = get_secret_stuff(data)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "use secret stuff":
            self.method = self.use_secret_stuff
            print("Euclidean distance method selected for evaluation")
        elif method_str.lower() == "demo_in_order":
            self.method = self.DEMO_IN_ORDER
            print("In order method selected for evaluation")
        else:
            raise Exception("Incorrect method selected for evaluation")

    def evaluate(self, population):
        start_timer("evaluation")
        eval = self.method(population)
        add_timer("evaluation")
        return eval

    def euclidean(population):
        """
        euclidean distance
        """
        pass

    # Need to use an efficient implementation for this
    def use_secret_stuff(self, population):
        distance = []
        for individual in population:
            current_distance = 0
            for a in range(len(population-1)):
                for b in range(a+1,len(population)):
                    small, big = individual[a], individual[b]
                    if small>big:
                        small, big = big, small
                    current_distance -= self.data[small,big]
            distance.append(current_distance)

        return np.array(distance)

    def DEMO_IN_ORDER(self, population):
        roll = np.roll(population, 1, axis=1)
        return (population[:,1:]> roll[:,1:] ).sum(axis=1)
