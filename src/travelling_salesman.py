"""
This is the top level module for the project.
The goal is to import functionality from a suite of modules to achieve modularity and quick prototyping.
As such this module should read only at a very high level of abstraction

Note: For the initial comit a number of demo methods have been written to show proposed workflow.
Anything with "DEMO" in it's name is obviously not a proper implementation and will be removed
"""

import numpy as np
from src.initialization import Initialization
from src.parent_selection import Parent_Selection
from src.recombination import Recombination
from src.mutation import Mutation
from src.evaluation import Evaluation
from src.survivor_selection import Survivor_Selection
import src.file_utils as files
from src.utils import *
from src.utils import debug_print as print

# CONSTS
POP_SIZE = 100
STR_LENGTH = 1000
NUM_PARENTS = 100
NUM_GENERATIONS = 100
MUTATION_RATE = .2
INIT_METHOD = "random_permutations"
SELECT_METHOD = "random"
CROSSOVER_METHOD = "cut_and_crossfill"
MUTATION_METHOD = "swap"
EVALUATION_METHOD = "DEMO_IN_ORDER"
SURVIVOR_METHOD = "mu_plus_lambda"
DEBUG = True


def DEMO_FUNCTIONALITY():
    """
    Provided in initial commit to demo workflow concept
    Begins with random permutations and evaluates fitness based on how many elements are increasing order
    eg) 2-3-4-3 has a fitness score of 3
    Yes, this is the worlds worst sorting algorithm.
    """
    # Initialize modules
    start_timer("setup")
    initializer = Initialization(POP_SIZE, STR_LENGTH, INIT_METHOD)
    parent_selector = Parent_Selection(POP_SIZE, STR_LENGTH, SELECT_METHOD)
    recombinator = Recombination(POP_SIZE, STR_LENGTH, CROSSOVER_METHOD)
    mutator = Mutation(str_length=STR_LENGTH, mutation_rate=MUTATION_RATE, method_str=MUTATION_METHOD)
    evaluator = Evaluation(graph=None, method_str=EVALUATION_METHOD)
    survivor_selector = Survivor_Selection(POP_SIZE, STR_LENGTH, SURVIVOR_METHOD)
    # Initialize Population
    population = initializer.initialize()
    fitness = evaluator.evaluate(population)
    population, mutation_index = mutator.mutate(population)
    end_timer("setup")

    # ITERATE FOR GENERATIONS
    print("*" * 20)
    print("Initial Mean Fitness: {}\t Best Fitness:{}".format(fitness.mean(), fitness.max()))
    print("Best initial member of Population:\n", population[np.argmax(fitness)])
    print("*" * 20)
    for n in range(NUM_GENERATIONS):
        start_timer("parent selection")
        parents = parent_selector.select(population, NUM_PARENTS)
        add_timer("parent selection")

        start_timer("recombination")
        children = recombinator.recombine(population, parents)
        add_timer("recombination")

        start_timer("mutation")
        children, children_mutation_index = mutator.mutate(children)
        population, population_mutation_index = mutator.mutate(population)
        add_timer("mutation")

        start_timer("evaluation")
        child_fitness = evaluator.evaluate(children)
        fitness = evaluator.evaluate(population)
        add_timer("evaluation")

        start_timer("survivor selection")
        population, fitness = survivor_selector.select(population, fitness, children, child_fitness)
        add_timer("survivor selection")

        # print debugs every 10% of the way
        if not (n % (NUM_GENERATIONS // 10)):
            print("Generation {}: Mean Fitness: {}\t Best Fitness:{}".format(n, fitness.mean(), fitness.max()))

    # finished, print results
    print("*" * 20)
    print("Final Mean Fitness: {}\t Best Fitness:{}".format(fitness.mean(), fitness.max()))
    print("Best Member of Population:\n", population[np.argmax(fitness)])

    print("*" * 10 + "\nFunction Times (in ms):\n")
    for k, v in get_times():
        print("{:16}\t{:.2f}".format(k, v * 1000))



if __name__ == '__main__':
    set_debug(DEBUG)
    DEMO_FUNCTIONALITY()
