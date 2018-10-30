"""
This is the top level module for the project.
The goal is to import functionality from a suite of modules to achieve modularity and quick prototyping.
As such this module should read only at a very high level of abstraction

Note: For the initial comit a number of demo methods have been written to show proposed workflow.
Anything with "DEMO" in it's name is obviously not a proper implementation and will be removed
"""

import numpy as np
import src.file_utils as files
from src.initialization import Initialization
from src.utils import debug_print as print, set_debug

# CONSTS
POP_SIZE = 10
STR_LENGTH = 5
INIT_METHOD = "DEMO_random"
DEBUG = True


def DEMO_FUNCTIONALITY():
    """
    Provided in initial commit to demo workflow concept
    """
    init = Initialization(POP_SIZE, STR_LENGTH, INIT_METHOD)
    population = init.initialize()
    print("Sample first 3 elements of the population:\n", population[:3])
    distances = files.parse_file("../data/TSP_WesternSahara_29.txt")
    print("Sample first 3 distances:\n")
    print(distances[:3])


def DEMO_DEBUG_PRINTING():
    """
    Provided in initial commit to demo debug printing
    """
    print("Demonstration of debug print functionality. The debug flag will be toggled off")
    set_debug(False)
    print("This line should not print")
    set_debug(DEBUG)


if __name__ == '__main__':
    set_debug(DEBUG)
    DEMO_FUNCTIONALITY()
    # DEMO_DEBUG_PRINTING()
