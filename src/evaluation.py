"""
Evaluation heuristics
"""
import numpy as np


class Evaluation:

    def __init__(self, graph, method_str):
        self.graph = graph
        self.set_method(method_str)

    def set_method(self, method_str):
        """
        Selects the method used by the Initialize function
        :param method_str:
        :return:
        """
        if method_str.lower() == "euclidean_distance":
            self.method = self.euclidean
            print("Euclidean distance method selected for evaluation")
        elif method_str.lower() == "demo_in_order":
            self.method = self.DEMO_IN_ORDER
            print("In order method selected for evaluation")
        else:
            raise Exception("Incorrect method selected for evaluation")

    def evaluate(self, population):
        return self.method(population)

    def euclidean(population):
        """
        euclidean distance
        """
        pass

    def DEMO_IN_ORDER(self, population):
        roll = np.roll(population, 1, axis=1)
        return (population[:,1:]> roll[:,1:] ).sum(axis=1)
