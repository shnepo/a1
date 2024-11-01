import numpy as np
from tsp import TSP
import matplotlib.pyplot as plt

class RandomSearch:
    def __init__(self, upper_limit, plot=False):
        """
        Initialize the RandomSearch class.

        Parameters:
            upper_limit (int): The max number of trials without improvement before stopping.
            plot (bool): Whether to plot the route at the end.
        """
        self.upper_limit = upper_limit
        self.plot = plot
        self.best_route = None
        self.best_distance = float('inf')
        self.convergence_history = []  # Track best distance over time for convergence

    def search(self):
        """Performs the random search for the best TSP route."""
        tsp = TSP(plot=False)
        no_improvement_counter = 0
        
        while no_improvement_counter < self.upper_limit:
            # Generate a random route
            random_route = np.random.permutation(tsp.dim)
            distance = tsp(random_route)

            # If the new route is better, update the best route
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = random_route
                no_improvement_counter = 0  # Reset counter if improvement is found
            else:
                no_improvement_counter += 1

            # Record the best distance at this step for convergence tracking
            self.convergence_history.append(self.best_distance)

        # Optionally plot the best route found
        if self.plot:
            self.plot_best_route(tsp)

        return self.best_route, self.best_distance

    def plot_best_route(self, tsp):
        """Plot the best route found using the TSP plotting utility."""
        with TSP(plot=True) as tsp_plot:
            tsp_plot.plot_route(tsp_plot.create_path(self.best_route), self.best_distance)

    def plot_convergence(self):
        """Plot the convergence history of the search."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, label="Random Search Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Best Distance Found")
        plt.title("Random Search Convergence Plot")
        plt.legend()
        plt.show()
