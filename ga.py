from tsp import *
import numpy as np

def swap_operator(path):
    mutation = np.random.choice(len(path), 2, replace = False) # Choosing two random gene indexes
    path[mutation[0]], path[mutation[1]] = path[mutation[1]], path[mutation[0]] # swapping
    return  path

def inversion_operator(path):
    """Inverse a random segment of the path/indivual/chromosome.
    Parameters:
        path (np.ndarray): one solution to problem
    Returns:
        path (np.ndarray): mutated path, with a random segment inverted
    """
    mutation = np.random.choice(len(path), 2, replace = False) 
    if mutation[0] > mutation[1]: # outsides of path being inversed 
        rightHalf = np.arange(mutation[0], len(path)) # creates array of numbers for remaining right half of array
        leftHalf = np.arange(0, mutation[1] + 1) # same for the left half up to lower index chosen
        mutationIdxs = np.append(rightHalf, leftHalf) # invert the two halves
    else: # inside of path being inversed
        mutationIdxs = np.arange(mutation[0],mutation[1])
    path[mutationIdxs] = path[mutationIdxs[::-1]]
    return path

def crossoverOperator(path0,path1):
    start = np.random.choice(len(path0))
    end = np.random.choice(range(start,len(path0)+1))
    crossoverString = path0[start:end] 
    path1 = path1[np.isin(path1, crossoverString, invert=True)]
    path1 = np.insert(path1, start, crossoverString)
    return path1

def geneticAlgotithm(nPaths=30,maxUnchangedIterations=1000, survivalRate=65, mutationRate=15, mutationOperator=inversion_operator):
    tsp = TSP()
    paths = np.array([np.random.permutation(tsp.dim) for _ in range(nPaths)])
    distances = np.array([tsp(path) for path in paths])
    minDist = np.min(distances)
    print(f"initial smallest distance = {minDist}\ninitiating genetic algorithm\n")
    unchangedCount = 0
    epoche = 1
    while unchangedCount < maxUnchangedIterations:
        # best % survives, unique produces the indexes of distances that produce the sorting of unique items, used to choose the surviving paths
        _, sortingIdxs = np.unique(distances, return_index=True)
        paths = paths[sortingIdxs][:nPaths*survivalRate//100]
        # teenage trurtles
        while len(paths)<nPaths*(survivalRate+mutationRate)/100:
            idx = np.random.choice(len(paths))
            mutatedPath = mutationOperator(paths[idx].copy())
            paths = np.append(paths, [mutatedPath], axis=0)
        #perform crossovers in the original surviving paths
        crossoverRate = 100 - (survivalRate+mutationRate)
        crossovers = [np.random.choice(nPaths*survivalRate//100,2, replace=False) for _ in range(nPaths*crossoverRate//100)]
        for crossover in  crossovers:
            crossoveredPath = crossoverOperator(paths[crossover[0]].copy(),paths[crossover[1]].copy())
            paths = np.append(paths, [crossoveredPath], axis = 0)
        #calculate new distances and keep track if improvements are being made
        distances = np.array([tsp(path) for path in paths])
        newMin = np.min(distances)
        if newMin < minDist:
            minDist = newMin
            unchangedCount = 0
        else:
            unchangedCount += 1
        #print status once every 50 epochs and update epoche
        if (not epoche%50):
            print(f'epoche {epoche}...')
            print(f'smallest distance = {newMin}')
        epoche += 1

    print(f'smallest distance found = {newMin}')
    best_route = np.argmin(distances)
    with TSP(plot=True) as tsp:
        tsp.plot_route(paths[best_route], distances[best_route])

geneticAlgotithm()
