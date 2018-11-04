"""
This is the top level module for the project.
The goal is to import functionality from a suite of modules to achieve modularity and quick prototyping.
As such this module should read only at a very high level of abstraction

Note: For the initial comit a number of demo methods have been written to show proposed workflow.
Anything with "DEMO" in it's name is obviously not a proper implementation and will be removed
"""

import numpy as np
from src.tsp_instance import TSP
from src.initialization import Initialization
from src.parent_selection import Parent_Selection
from src.recombination import Recombination
from src.mutation import Mutation
from src.evaluation import Evaluation
from src.survivor_selection import Survivor_Selection
import src.file_utils as files
from src.utils import *
from src.utils import debug_print as print
from src.preprocessing import get_secret_stuff
from src.file_utils import parse_file as parse
import matplotlib.pyplot as plt

# CONSTS
POP_SIZE = 20
STR_LENGTH = 10
NUM_PARENTS = 10
NUM_GENERATIONS = 50
MUTATION_RATE = .2
INIT_METHOD = "random_permutations"
SELECT_METHOD = "random"
CROSSOVER_METHOD = "cut_and_crossfill"
MUTATION_METHOD = "swap"
EVALUATION_METHOD = "use secret stuff"
SURVIVOR_METHOD = "mu_plus_lambda"
DEBUG = True


def plot(data, arr):
    #plt.scatter(*zip(*data))

    x,y = zip(*data)

    for i in range(0, len(arr)-1):
        plt.plot(x[arr[i]:arr[i+1]], y[arr[i]:arr[i + 1]], 'ro-')
    plt.show()


def DEMO_FUNCTIONALITY():
    """
    Provided in initial commit to demo workflow concept
    """
    # Initialize modules
    start_timer("setup")
    big_data = "../data/TSP_Canada_4663.txt"
    middle_data = "../data/TSP_Uruguay_734.txt"
    small_data = "../data/TSP_WesternSahara_29.txt"
    actual_data = parse(small_data)
    # Create Instance
    tsp = TSP(
        graph           = actual_data,
        population_size = POP_SIZE,
        num_parents     = NUM_PARENTS,
        mutation_rate   = MUTATION_RATE,
        num_generations = NUM_GENERATIONS
    )
    # Initialize modules
    initializer = Initialization(tsp, INIT_METHOD)
    parent_selector = Parent_Selection(tsp, SELECT_METHOD)
    recombinator = Recombination(tsp, CROSSOVER_METHOD)
    mutator = Mutation(tsp, method_str=MUTATION_METHOD)
    evaluator = Evaluation(tsp, method_str=EVALUATION_METHOD)
    survivor_selector = Survivor_Selection(tsp, SURVIVOR_METHOD)
    # Initialize Population and fitness
    initializer.initialize()
    evaluator.evaluate()
    end_timer("setup")

    # ITERATE FOR GENERATIONS
    print("*" * 20)
    print("Initial Mean Fitness: {}\t Best Fitness:{}".format(tsp.fitness.mean(), tsp.fitness.max()))
    print("Best initial member of Population:\n", tsp.population[np.argmax(tsp.fitness)])
    print("*" * 20)
    for n in range(NUM_GENERATIONS):
        # select parents and spawn children
        parent_selector.select()
        recombinator.recombine()
        # mutate population and children
        mutator.mutate_population()
        mutator.mutate_children()
        # re-evaluate children and population
        evaluator.evaluate()
        evaluator.evaluate_children()
        # select from parents and children to form new population
        survivor_selector.select()
        # add history and print debugs every 10%
        tsp.add_history("mean_fitness",tsp.fitness.mean())
        tsp.add_history("best_fitness",tsp.fitness.max())
        if not (n % (NUM_GENERATIONS // 10)):
            print("Generation {:<4} Mean Fitness: {:5.2f}\t Best Fitness:{}".format(n, tsp.fitness.mean(), tsp.fitness.max()))

    # finished, print results
    print("*" * 20)
    print("Final Mean Fitness: {}\t Best Fitness:{}".format(tsp.fitness.mean(), tsp.fitness.max()))
    print("Best Member of Population:\n", tsp.population[np.argmax(tsp.fitness)])
    print("*" * 10 + "\nFunction Times (in ms):\n")
    for k, v in get_times():
        print("{:16}\t{:.2f}".format(k, v * 1000))

    tsp.plot_history("mean_fitness")
    tsp.plot_history("best_fitness")


if __name__ == '__main__':
    set_debug(DEBUG)
    DEMO_FUNCTIONALITY()
