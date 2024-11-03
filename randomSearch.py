import numpy as np
from tsp import TSP
import matplotlib.pyplot as plt

class RandomSearch:
    def __init__(self, upper_limit=10000, plot=False, endParameter='epoch'):
        """
        Initialize the RandomSearch class.

        Parameters:
            upper_limit (int): The max number of trials without improvement before stopping.
            plot (bool): Whether to plot the route at the end.
        """
        self.plot = plot
        self.end = (lambda: self.epoch>=upper_limit) if endParameter=='epoch' else (lambda: self.no_improvement_counter>=upper_limit) 
        self.best_route = None
        self.best_distance = float('inf')
        self.convergence_history = []  # Track best distance over time for convergence
        self.epoch = 0
        self.no_improvement_counter = 0

    def __call__(self):
        """Performs the random search for the best TSP route."""
        tsp = TSP(plot=False)     
        while not self.end():
            # Generate a random route
            random_route = np.random.permutation(tsp.dim)
            distance = tsp(random_route)

            # If the new route is better, update the best route
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = random_route
                self.no_improvement_counter = 0  # Reset counter if improvement is found
            else:
                self.no_improvement_counter += 1
            self.epoch += 1
            # Record the best distance at this step for convergence tracking
            self.convergence_history.append(self.best_distance)

        # Optionally plot the best route found
        if self.plot:
            self.plot_best_route(tsp)

    def plot_best_route(self, tsp):
        """Plot the best route found using the TSP plotting utility."""
        with TSP(plot=True) as tsp_plot:
            tsp_plot.plot_route(tsp_plot.create_path(self.best_route), self.best_distance)

    def plot_convergence(self, label="Random Search Convergence"):
        """Plot the convergence history of the search."""
        plt.plot(self.convergence_history, label=label)
