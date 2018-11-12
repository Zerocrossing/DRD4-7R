"""
This is the top level module for the project.
The goal is to import functionality from a suite of modules to achieve modularity and quick prototyping.
As such this module should read only at a very high level of abstraction
"""

import numpy as np
from src.tsp_instance import TSP
from src.initialization import Initialization
from src.parent_selection import Parent_Selection
from src.recombination import Recombination
from src.mutation import Mutation
from src.evaluation import Evaluation
from src.survivor_selection import Survivor_Selection
from src.termination import Termination
from src.animation import Animation
import src.file_utils as files
from src.utils import *
from src.utils import debug_print as print
from src.file_utils import parse_file as parse

# CONSTS
POP_SIZE = 1000
NUM_PARENTS = 100
NUM_GENERATIONS = 500
TIME_LIMIT = 100000
MUTATION_RATE = .3
INIT_METHOD = "random_permutations"
# SELECT_METHOD = "random"
SELECT_METHOD = "mu_plus_lambda"
# CROSSOVER_METHOD = "cut_and_crossfill"
CROSSOVER_METHOD = "order_crossover"
# MUTATION_METHOD = "swap"
MUTATION_METHOD = "flip"
EVALUATION_METHOD = "cached_euclidean"
# SURVIVOR_METHOD = "random"
SURVIVOR_METHOD = "mu_plus_lambda"
TERMINATOR_METHOD = "num_iterations"
DEBUG = True
PLOT = False


def the_tsp_problem():
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
        num_generations = NUM_GENERATIONS)
    # Initialize modules
    initializer         = Initialization(tsp, INIT_METHOD)
    parent_selector     = Parent_Selection(tsp, SELECT_METHOD)
    recombinator        = Recombination(tsp, CROSSOVER_METHOD)
    mutator             = Mutation(tsp, MUTATION_METHOD)
    evaluator           = Evaluation(tsp, EVALUATION_METHOD)
    survivor_selector   = Survivor_Selection(tsp, SURVIVOR_METHOD)
    terminator          = Termination(NUM_GENERATIONS, TIME_LIMIT, TERMINATOR_METHOD)
    animator            = Animation(actual_data)
    # Initialize Population and fitness
    initializer.initialize()
    evaluator.evaluate()
    end_timer("setup")
    print("*" * 20)
    print("Initial Mean Fitness: {}\t Best Fitness:{}".format(tsp.fitness.mean(), tsp.fitness.max()))
    # print("Best initial member of Population:\n", tsp.population[np.argmax(tsp.fitness)])
    print("*" * 20)
    current_time = 0

    while terminator.method(tsp.current_generation, current_time):
        # select parents and spawn children
        parent_selector.select()
        recombinator.recombine()
        # mutate population and children
        mutator.mutate_population()
        mutator.mutate_children()
        # re-evaluate children and population
        evaluator.evaluate(use_mask=True)
        evaluator.evaluate_children()
        # select from parents and children to form new population
        survivor_selector.select()
        # add history and print debugs every 10%
        tsp.add_history("mean_fitness",tsp.fitness.mean())
        tsp.add_history("best_fitness",tsp.fitness.max())
        tsp.add_history("best_individual",tsp.population[np.argmax(tsp.fitness.max())])

        tsp.current_generation +=1
        if not (tsp.current_generation % (tsp.num_generations // 10)):
            # print("Mutation Rate:",tsp.mutation_rate)
            print("Generation {:<4} Mean Fitness: {:5.2f}\t Best Fitness:{}".format(tsp.current_generation, tsp.fitness.mean(), tsp.fitness.max()))
            if PLOT:
                animator.update(tsp.population[np.argmax(tsp.fitness.max())])

    # finished, print results
    print("*" * 20)
    # print("Best Member of Population:\n", tsp.population[np.argmax(tsp.fitness)])
    print("Final Mean Fitness: {}\t Best Fitness:{}".format(tsp.fitness.mean(), tsp.fitness.max()))
    print("*" * 10 + "\nFunction Times (in ms):\n")
    time_sum = 0
    for k, v in get_times():
        print("{:16}\t{:.2f}".format(k, v * 1000))
        time_sum+=v
    print("-"*20)
    print("Total Time:\t{:.2f} seconds".format(time_sum))

    if PLOT:
        animator.update(tsp.population[np.argmax(tsp.fitness.max())])

    # plot history
    tsp.plot_history("mean_fitness")
    tsp.plot_history("best_fitness")



if __name__ == '__main__':
    set_debug(DEBUG)
    the_tsp_problem()