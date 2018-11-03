import numpy as np

# crude start to preprocessing

def euclidean(a,b):
    return np.linalg.norm(a-b)

def main():

    with open("../data/TSP_Uruguay_734.txt") as f:
        data = f.readlines()

    population= []
    for line in data:
        temp = line.split()
        population.append([int(temp[0]), float(temp[1]), float(temp[2])])

    distances = {}

    for a in range(len(population)-1):
        for b in range(a+1, len(population)):
            key = (population[a][0], population[b][0])
            point_a , point_b = np.array(population[a][1:]), np.array(population[b][1:])
            distances[key] = euclidean(point_a, point_b)

if __name__ == '__main__':
    main()

