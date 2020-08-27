# DRD4-7R: _The Explorer Gene_
*A genetic algorithm which attempts to solve the traveling salesman problem.*

By Bishop, Robert; Elsisy, Moustafa; Graves, Laura; and Nagisetty, Vineel: all affiliated with the Department of Computer Science at Memorial University of Newfoundland

## Table of Contents
* Introduction
* Usage
* More Information

## Introduction:
Our research aims to solve the travelling salesman problem using data found [here](http://www.math.uwaterloo.ca/tsp/world/countries.html#WI).  We plan to use an evolutionary computing algorithm to solve this.

* Summary of operators used:

| Feature               | Operators Used                   |
|-----------------------|:--------------------------------:|
| Recombination         | Order, PMX, SCX                  |
| Recombination %       | 100%                             |
| Mutation              | Swap, Flip (inversion), Scramble |
| Mutation %            | 20%                              |
| Parent selection      | Roulette Wheel with Elitism      |
| Survival selection    | Mu + Lambda                      |
| Population size       | 200                              |
| Offsprings            | 100                              |
| Initialisation        | Random                           |
| Termination condition | 10,000 generations               |

* Code Optimizations: 
  * Pre-calculating distances between cities and storing the values to a lookup table prior to initialisation
  * Parallelizing pre-calculation and using a just in time compiler to speed up execution
  * Using 16-bit unsigned integers to reduce memory usage by a quarter compared to regular 64 bit integers
  * An animation of the best solution generated every set number of generations is provided for a more intuitive visualization of the solution.
 
 ## Usage:
 ### Source Code:
 * The main module to run is the `travelling_salesman.py` file and is found in the `src/` directory
 * The rest of the modules in the `src/` directory implement the different features of an evolutionary algorithm
 * The Data used to run the code on is found in the `data/` directory
 
 ### Requirements:
 * numba
 * jupyter
 * numpy
 * matplotlib
 * ffmpeg
 
 ### Reproduce Results:
 * To reproduce the results shown in the report, please run the notebook file `Report Viz.ipynb` found in the main directory. 
 * To view the animation, run the `travelling_salesman.py` module, setting `ANIMATION = True`
 * To view the debug information in the `travelling_saleslan.py` module, set `DEBUG = True` 
 
 ## More Information: 
 - More information can be found [here](https://github.com/Zerocrossing/DRD4-7R/blob/master/3201_Explorer_Gene.pdf)
