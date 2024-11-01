from randomSearch import RandomSearch as RS
from ga import GeneticAlgorithm as GA
import matplotlib.pyplot as plt

def run_random_search_experiment(upper_limit):
    """Runs the Random Search experiment and plots the results."""
    print("Starting Random Search Experiment...")
    random_search = RS(upper_limit, plot=True)
    best_route, best_distance = random_search.search()
    print(f"Random Search - Best Distance: {best_distance:.2f} km")
    random_search.plot_convergence()

def run_ga_experiment(mutation_operator):
    """Runs the Genetic Algorithm experiment with a specified mutation operator and plots the results."""
    print(f"Starting GA Experiment with {mutation_operator} mutation...")
    ga = GA(n_paths=30, max_unchanged_iterations=1000, mutation_operator=mutation_operator)
    best_route, best_distance = ga.optimize()
    print(f"GA ({mutation_operator}) - Best Distance: {best_distance:.2f} km")
    ga.plot_convergence()

if __name__ == "__main__":
    # Run Random Search Experiment
    upper_limit = 100000 # amount of times RS continues to try without an improvement being found.
    run_random_search_experiment(upper_limit)
    
    # Run GA Experiment with Swap Mutation
    run_ga_experiment(mutation_operator='swap')
    
    # Run GA Experiment with Inversion Mutation
    run_ga_experiment(mutation_operator='inversion')
