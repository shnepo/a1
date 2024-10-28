import numpy as np
from tsp import TSP 

def random_search(upper_limit):
    """Performs a random search for the best TSP route.

    Parameters
    upper_limit: int
        Hyperparameter that indicates how many times the algorithm will continue searching for a
        better route before deciding to stop.

    Returns
    tuple
        Best_route: (route(set), distance_of_route(int))
    """
    # Initialize the TSP instance
    tsp = TSP(plot=False)
    
    
    best_route = (None, float('inf'))
    total_routes_since_last_update = 0

    # keeps going until upper limit is reached
    while True:
        # generates a random route
        random_route = np.random.permutation(tsp.dim)

        # evaluates the random route
        distance_random_route = tsp(random_route)

        current_route = (random_route, distance_random_route)

        # check if current route is shorter
        if current_route[1] < best_route[1]:
            best_route = current_route
            total_routes_since_last_update = 0
            continue  

        total_routes_since_last_update += 1

        # check if we reached upper limit
        if total_routes_since_last_update >= upper_limit:
            break 

    return best_route

if __name__ == "__main__":
    upper_limit = 100  # set upper limit
    best_route = random_search(upper_limit)
    print(f"Best route: {best_route[0]}, Distance: {best_route[1]:.2f} km")
