import numpy as np
from tsp import TSP
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, n_paths=30, max_unchanged_iterations=1000, survival_rate=65, mutation_rate=15, mutation_operator='swap'):
        """
        Initialize the GeneticAlgorithm class.

        Parameters:
            n_paths (int): Number of paths (individuals) in the population.
            max_unchanged_iterations (int): Max iterations without improvement before stopping.
            survival_rate (int): Percentage of population retained as parents.
            mutation_rate (int): Percentage of population to mutate.
            mutation_operator (str): Mutation operator to use ('swap' or 'inversion').
        """
        self.n_paths = n_paths
        self.max_unchanged_iterations = max_unchanged_iterations
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.mutation_operator = self.swap_operator if mutation_operator == 'swap' else self.inversion_operator
        self.best_distance = float('inf')
        self.best_route = None
        self.convergence_history = []

    def swap_operator(self, path):
        """Mutates a path by swapping two random cities."""
        mutation = np.random.choice(len(path), 2, replace=False)
        path[mutation[0]], path[mutation[1]] = path[mutation[1]], path[mutation[0]]
        return path

    def inversion_operator(self, path):
        """Mutates a path by inverting a random segment."""
        mutation = np.random.choice(len(path), 2, replace=False)
        start, end = min(mutation), max(mutation)
        path[start:end] = path[start:end][::-1]
        return path

    def crossover_operator(self, parent1, parent2):
        """Performs ordered crossover between two parent paths."""
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer == len(child):
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        return np.array(child)

    def optimize(self):
        """Runs the genetic algorithm optimization process for TSP."""
        tsp = TSP(plot=False)
        # Initialize population with random paths
        population = np.array([np.random.permutation(tsp.dim) for _ in range(self.n_paths)])
        distances = np.array([tsp(path) for path in population])
        
        # Set initial best path
        self.best_distance = distances.min()
        self.best_route = population[distances.argmin()]
        unchanged_iterations = 0
        
        while unchanged_iterations < self.max_unchanged_iterations:
            # Track convergence
            self.convergence_history.append(self.best_distance)
            
            # Selection: Retain top performers
            sorted_indices = np.argsort(distances)
            population = population[sorted_indices][:self.n_paths * self.survival_rate // 100]
            
            # Mutation
            while len(population) < self.n_paths * (self.survival_rate + self.mutation_rate) / 100:
                individual = self.mutation_operator(population[np.random.randint(len(population))].copy())
                population = np.vstack([population, individual])
            
            # Crossover
            while len(population) < self.n_paths:
                parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]
                child = self.crossover_operator(parent1, parent2)
                population = np.vstack([population, child])
            
            # Update distances and check if we improved
            distances = np.array([tsp(path) for path in population])
            current_best_distance = distances.min()
            
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_route = population[distances.argmin()]
                unchanged_iterations = 0
            else:
                unchanged_iterations += 1

        # Optionally plot the best route found
        self.plot_best_route(tsp)
        return self.best_route, self.best_distance

    def plot_best_route(self, tsp):
        """Plots the best route found on the map of Europe."""
        with TSP(plot=True) as tsp_plot:
            tsp_plot.plot_route(tsp_plot.create_path(self.best_route), self.best_distance)

    def plot_convergence(self):
        """Plot the convergence history of the GA search."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, label="Genetic Algorithm Convergence")
        plt.xlabel("Generations")
        plt.ylabel("Best Distance Found")
        plt.title("Genetic Algorithm Convergence Plot")
        plt.legend()
        plt.show()

