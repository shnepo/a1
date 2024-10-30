from tsp import *
import numpy as np

tsp = TSP()
n_paths = 50
paths = np.array([np.random.permutation(tsp.dim) for _ in range(n_paths)])
distances = np.array([tsp(path) for path in paths])
min_dist = np.min(distances)
print(f"initial smallest distance = {min_dist}\ninitiating genetic algorithm\n")

def swap_operator(path):
    mutation = np.random.choice(len(path), 2, replace = False)
    path[mutation[0]], path[mutation[1]] = path[mutation[1]], path[mutation[0]]
    return  path

def inversion_operator(path):
    mutation = np.random.choice(len(path), 2, replace = False)
    if mutation[0]>mutation[1]:
        right_half = np.arange(mutation[0], len(path))
        left_half = np. arange(0, mutation[1]+1)
        mutation_idxs = np.append(right_half, left_half)
    else:
        mutation_idxs = np.arange(mutation[0],mutation[1])
    path[mutation_idxs] = path[mutation_idxs[::-1]]
    return path

    

unchanged_count = 0
epoche = 1
while unchanged_count < 1000:
    # best % survives
    survival_rate = 65
    _, sorting_idxs = np.unique(distances, return_index=True)
    paths = paths[sorting_idxs][:n_paths*survival_rate//100]
    mutation_rate = 15 
    mutation_operator = inversion_operator
    # teenage trurtles
    while len(paths)<n_paths*(survival_rate+mutation_rate)/100:
        idx = np.random.choice(len(paths))
        path_clone = paths[idx].copy()
        path_clone = mutation_operator(path_clone)
        paths = np.append(paths, [path_clone], axis=0)

    crossover_rate = 100 - (survival_rate+mutation_rate)
    #perform crossovers in the original surviving paths
    crossovers = [np.random.choice(n_paths*survival_rate//100,2, replace=False) for _ in range(n_paths*crossover_rate//100)]
    for crossover in  crossovers:
        path0 = paths[crossover[0]].copy()
        path1 = paths[crossover[1]].copy()
        start = np.random.choice(len(path0))
        end = np.random.choice(range(start,len(path0)+1))
        crossover_string = path0[start:end] 
        path2 = path1.copy()
        path2 = path2[np.isin(path2, crossover_string, invert=True)]
        path2 = np.insert(path2, start, crossover_string)
        paths = np.append(paths, [path2], axis = 0)
    distances = np.array([tsp(path) for path in paths])
    new_min = np.min(distances)
    if not epoche%50:
        print(f'epoche {epoche}...')
        print(f'smallest distance = {new_min}')
    if new_min < min_dist:
        min_dist = new_min
        unchanged_count = 0
    else:
        unchanged_count += 1
    epoche += 1

print(f'smallest distance found = {new_min}')
best_route = np.argmin(distances)
with TSP(plot=True) as tsp:
    tsp.plot_route(paths[best_route], distances[best_route])

