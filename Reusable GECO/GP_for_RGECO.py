from deap import tools, base, creator
from deap.gp import compile, PrimitiveSetTyped, PrimitiveTree, genHalfAndHalf, graph, cxOnePoint, mutUniform, staticLimit
import numpy as np
import operator as op
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import functools
import time
import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyTorch model functions instead of TensorFlow
from PyTorch.GPfunctions import get_ga_fitness, get_model_weights, get_fitness
# Import GA function to get dataset
from GA_without_crossover import run_genetic_algorithm

# GPU setup for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize dataset as None
dataset = None

def initialize_dataset():
    """Initialize the dataset by running the GA if not already done"""
    global dataset
    if dataset is None:
        print("Getting dataset from Genetic Algorithm...")
        dataset = run_genetic_algorithm()
        print(f"Dataset received from GA with {len(dataset)} pairs")
    return dataset

# Model parameters - same as original but now using PyTorch
parent_length = 12227  # Hard coded for testing, will be calculated from PyTorch model

"""
GP Parameters from the paper: 

    Parameter                               Value
    Population size                          15
    Mutation rate                            0.3
    Crossover rate                           0.6
    Elitism rate                             0.1
    Tournament size                          2
    Number of generations (Reusable)         150
    Number of generations (Disposable)       10
"""
MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 4
MAX_TREE_SIZE = 20 # This controls the number of nodes in the tree to reduce bloat
POPULATION_SIZE = 15 
GENERATIONS = 10 
ELITISM_RATE = 0.1
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.6

# NEW: Maximum attempts for generating valid trees to prevent infinite loops
MAX_GENERATION_ATTEMPTS = 100

"""
TERMINAL SET and PRIMITIVE SET remain the same as they operate on numpy arrays
which are compatible with both TensorFlow and PyTorch after conversion
"""

# NEW: Helper function to check tree validity
def is_tree_valid(tree):
    """
    Check if a tree meets size and depth constraints.
    Returns True if valid, False otherwise.
    """
    try:
        return len(tree) <= MAX_TREE_SIZE and tree.height <= MAX_TREE_DEPTH
    except (IndexError, AttributeError):
        return False

# NEW: Improved tree generation with size control
def generate_constrained_tree(pset, min_depth, max_depth, max_size):
    """
    Generate a single tree that satisfies both depth and size constraints.
    Uses multiple attempts to avoid infinite loops.
    """
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        # Start with smaller depth and gradually increase if needed
        current_max_depth = min(min_depth + (attempt // 10), max_depth)
        
        try:
            tree = genHalfAndHalf(pset, min_depth, current_max_depth)
            individual = creator.Individual(tree)
            
            if is_tree_valid(individual):
                return individual
        except Exception as e:
            # If generation fails, try again
            continue
    
    # If all attempts fail, create a minimal valid tree
    print("Warning: Creating minimal tree due to generation failures")
    return create_minimal_tree(pset)

# NEW: Create minimal valid tree as fallback
def create_minimal_tree(pset):
    """
    Create a minimal valid tree when all generation attempts fail.
    This ensures we never get stuck in infinite loops.
    """
    # Create a simple tree that just returns the sum of the two parents
    from deap.gp import Terminal, Primitive
    
    # Create a simple tree: Sum(P1, P2)
    tree = creator.Individual([
        Primitive('Sum', [np.ndarray, np.ndarray], np.ndarray),
        Terminal('P1', True, np.ndarray),
        Terminal('P2', True, np.ndarray)
    ])
    return tree

"""
OLD: Since DEAP does not offer bloat control natively during initial population generation we gots to design our own population generation fucntion to ensure that we dont generate trees with too many node
NEW: Completely rewritten population generation with proper bloat control and fallback mechanisms
"""
def generate_bloatless_population(n):
    """
    NEW: Generate initial population with both size and height constraints.
    Improved version that prevents infinite loops and ensures all trees are valid.
    """
    population = []
    failed_attempts = 0
    max_failed_attempts = n * 10  # Allow some failures but not infinite
    
    print(f"Generating population of {n} individuals with max size {MAX_TREE_SIZE} and max depth {MAX_TREE_DEPTH}")
    
    while len(population) < n and failed_attempts < max_failed_attempts:
        try:
            # NEW: Use constrained tree generation instead of basic toolbox.individual()
            ind = generate_constrained_tree(pset, MIN_TREE_DEPTH, MAX_TREE_DEPTH, MAX_TREE_SIZE)
            
            if is_tree_valid(ind):
                population.append(ind)
                print(f"Generated individual {len(population)}/{n} - Size: {len(ind)}, Depth: {ind.height}")
            else:
                failed_attempts += 1
                
        except Exception as e:
            failed_attempts += 1
            print(f"Failed to generate individual: {e}")
    
    # NEW: If we couldn't generate enough individuals, fill with minimal trees
    while len(population) < n:
        minimal_tree = create_minimal_tree(pset)
        population.append(minimal_tree)
        print(f"Added minimal tree {len(population)}/{n}")
    
    print(f"Population generation complete. Generated {len(population)} individuals.")
    return population


def vector_sum(parent1, parent2):
    """Element-wise sum of two weight vectors"""
    return parent1 + parent2

def vector_mean(parent1, parent2):
    """Element-wise mean of two weight vectors"""
    return (parent1 + parent2) / 2

def one_point_crossover(i_point: int, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """
    One-point crossover operation for weight vectors.
    Works the same regardless of whether weights come from TensorFlow or PyTorch.
    """
    i_point = int(i_point)
    return np.concatenate([w1[:i_point], w2[i_point:]])

def compare_mean_less(w1: np.ndarray, w2: np.ndarray) -> bool:
    """Compare mean values of two weight vectors"""
    return np.mean(w1) < np.mean(w2)

def compare_norm_greater(w1: np.ndarray, w2: np.ndarray) -> bool:
    """Compare L2 norms of two weight vectors"""
    return np.linalg.norm(w1) > np.linalg.norm(w2)

def select_fitness(condition, F1, F2):
    """Select fitness value based on condition"""
    return F1 if condition else F2

def select_weights(condition, W1, W2):
    """Select weight vector based on condition"""
    return W1 if condition else W2

def get_random_crossover_point():
    """Generate random crossover point"""
    return random.randint(1, parent_length - 1)

def get_gp_fitness(individual):
    """
    Evaluate GP individual fitness using PyTorch models.
    
    Key improvements over TensorFlow version:
    1. GPU acceleration for neural network evaluation
    2. More efficient tensor operations
    3. Better memory management with PyTorch
    4. Potentially faster training with optimized CUDA operations
    """
    # Ensure dataset is initialized
    global dataset
    if dataset is None:
        dataset = initialize_dataset()
    
    # Compile the GP tree into a callable function
    func = toolbox.compile(individual)
    fitness = 0
    
    print(f"Evaluating GP individual on {len(dataset)} data points...")

    for i, row in dataset.iterrows():
        parent1_data = row['parent1']
        parent2_data = row['parent2']
        
        # Extract weights and fitness from GA dataset format
        parent1 = parent1_data['parent1'][0]  # Get the numpy array of weights
        p1_fitness = parent1_data['fitness']
        parent2 = parent2_data['parent2'][0]  # Get the numpy array of weights
        p2_fitness = parent2_data['fitness']
        
        try:
            # Call the compiled tree function to generate offspring
            offspring = func(parent1, parent2, p1_fitness, p2_fitness)
            
            # Evaluate offspring fitness using PyTorch model (GPU accelerated)
            offspring_fitness = get_ga_fitness(offspring)
            
            # Calculate fitness based on parent and offspring performance
            parent_avg_fitness = (p1_fitness + p2_fitness) / 2
            fitness += float(parent_avg_fitness / offspring_fitness)
            
        except Exception as e:
            print(f"Error evaluating individual on datapoint {i}: {e}")
            """
            OLD: Earlier this was float('inf') which gave the error based trees the maximum fitness
            this was a big mistake as we are maximing the GP fitness and selecting the individual
            with the highest fitness , so this mistake is fixed
            """
            return float('-inf')  # Return worst possible fitness for invalid trees
    
    final_fitness = fitness / len(dataset)
    print(f"GP individual fitness: {final_fitness}")
    return final_fitness

# NEW: Improved safe crossover with better bloat control
def safe_mate(ind1, ind2):
    """
    NEW: Improved safe wrapper for mating operation with strict bloat control.
    Ensures offspring meet size and depth constraints or returns parents.
    """
    # Check if parents are valid for mating
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    # Try multiple crossover attempts with different points
    for attempt in range(5):  # NEW: Multiple attempts for better success rate
        try:
            # Create copies to avoid modifying originals
            parent1_copy = creator.Individual(ind1)
            parent2_copy = creator.Individual(ind2)
            
            # Perform the crossover
            offspring1, offspring2 = cxOnePoint(parent1_copy, parent2_copy)

            # NEW: Strict validation - both offspring must be valid
            if (is_tree_valid(offspring1) and is_tree_valid(offspring2) and 
                len(offspring1) > 0 and len(offspring2) > 0):
                
                print(f"Successful crossover - Offspring sizes: {len(offspring1)}, {len(offspring2)}")
                return offspring1, offspring2
            
        except Exception as e:
            print(f"Crossover attempt {attempt + 1} failed: {e}")
            continue
    
    # NEW: If all crossover attempts fail, return parents (no change)
    print("All crossover attempts failed, returning parents")
    return ind1, ind2

# NEW: Improved safe mutation with better bloat control
def safe_mutate(ind):
    """
    NEW: Improved safe wrapper for mutation operation with strict bloat control.
    Ensures mutated tree meets size and depth constraints or returns original.
    """
    if len(ind) < 2:
        return ind,
    
    # Try multiple mutation attempts
    for attempt in range(5):  # NEW: Multiple attempts for better success rate
        try:
            # Create copy to avoid modifying original
            ind_copy = creator.Individual(ind)
            
            # NEW: Use smaller subtrees for mutation to reduce bloat
            mutant = mutUniform(ind_copy, expr=toolbox.expr_mut, pset=pset)[0]
            
            # NEW: Strict validation
            if is_tree_valid(mutant) and len(mutant) > 0:
                print(f"Successful mutation - New size: {len(mutant)}")
                return mutant,
                
        except Exception as e:
            print(f"Mutation attempt {attempt + 1} failed: {e}")
            continue
    
    # NEW: If all mutation attempts fail, return original (no change)
    print("All mutation attempts failed, returning original")
    return ind,

# OLD: Dual static limit decorator
# NEW: Enhanced dual static limit with better error handling and logging
def dual_static_limit(max_height, max_size):
    """
    NEW: Enhanced decorator that enforces both height and size limits.
    Improved error handling and logging for better debugging.
    """
    def decorator(operator):
        def wrapper(*args, **kwargs):
            offspring = operator(*args, **kwargs)
            valid_offspring = []
            
            # For crossover, args[0] and args[1] are parents
            # For mutation, args[0] is the parent
            parents = args[:2] if len(args) > 1 else [args[0]]
            
            for i, child in enumerate(offspring):
                try:
                    if is_tree_valid(child):  # NEW: Use centralized validation
                        valid_offspring.append(child)
                        print(f"Valid offspring {i}: size={len(child)}, height={child.height}")
                    else:
                        # If child is invalid, use the corresponding parent
                        parent_idx = i % len(parents)
                        parent_copy = creator.Individual(parents[parent_idx])
                        valid_offspring.append(parent_copy)
                        print(f"Invalid offspring {i} replaced with parent: size={len(parent_copy)}")
                        
                except Exception as e:
                    # If validation fails, use the corresponding parent
                    parent_idx = i % len(parents)
                    parent_copy = creator.Individual(parents[parent_idx])
                    valid_offspring.append(parent_copy)
                    print(f"Offspring {i} validation error, replaced with parent: {e}")
            
            return tuple(valid_offspring)
        return wrapper
    return decorator


# Set up DEAP primitives and terminals (same as original)
pset = PrimitiveSetTyped("main", [np.ndarray, np.ndarray, float, float], np.ndarray)
pset.addTerminal(1, float)
pset.addTerminal(2, float)
pset.addTerminal(0.5, float)
pset.addTerminal(1, bool)
pset.addTerminal(0, bool)

# Add primitives (same as original)
pset.addPrimitive(op.add, [float, float], float)
pset.addPrimitive(op.sub, [float, float], float)
pset.addPrimitive(op.mul, [float, float], float)
pset.addPrimitive(vector_sum, [np.ndarray, np.ndarray], np.ndarray, name="Sum")
pset.addPrimitive(vector_mean, [np.ndarray, np.ndarray], np.ndarray, name="Mean")
pset.addPrimitive(op.lt, [float, float], bool)
pset.addPrimitive(op.gt, [float, float], bool)
pset.addPrimitive(compare_mean_less, [np.ndarray, np.ndarray], bool, name="weight_mean_lt")
pset.addPrimitive(compare_norm_greater, [np.ndarray, np.ndarray], bool, name="weight_gt")
pset.addPrimitive(select_weights, [bool, np.ndarray, np.ndarray], np.ndarray, name="Select_weights")
pset.addPrimitive(select_fitness, [bool, float, float], float, name="Select_fitness")

# Add ephemeral constant and one-point crossover
pset.addEphemeralConstant("fixed_crossover_point", functools.partial(get_random_crossover_point), int)
pset.addPrimitive(one_point_crossover, [int, np.ndarray, np.ndarray], np.ndarray, name="OnePointCrossover")

# Rename arguments for clarity
pset.renameArguments(ARG0="P1", ARG1="P2", ARG2="F1", ARG3="F2")

# Create DEAP types and toolbox (same as original)
"""
Fitness Max since here we are evaluating the GP with the highest fitness ratio
"""
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
# NEW: Use constrained tree generation for both initial and mutation trees
toolbox.register("expr", lambda: generate_constrained_tree(pset, MIN_TREE_DEPTH, MAX_TREE_DEPTH, MAX_TREE_SIZE))
toolbox.register("individual", lambda: toolbox.expr())  # NEW: Simplified individual creation
toolbox.register("population", generate_bloatless_population)
toolbox.register("compile", compile, pset=pset)
toolbox.register("evaluate", get_gp_fitness)
# NEW: Smaller trees for mutation to reduce bloat
# FIXED: Added **kwargs to handle all keyword arguments passed by mutUniform
toolbox.register("expr_mut", lambda pset, **kwargs: generate_constrained_tree(pset, MIN_TREE_DEPTH, min(MAX_TREE_DEPTH-1, 3), MAX_TREE_SIZE//2))
toolbox.register("select", tools.selTournament, k=1, tournsize=3, fit_attr='fitness')

# NEW: Register improved safe operations
toolbox.register("mate", safe_mate)
toolbox.register("mutate", safe_mutate)

# OLD: Add decorators for tree height control
# NEW: Use enhanced dual static limit decorator
toolbox.decorate("mate", dual_static_limit(MAX_TREE_DEPTH, MAX_TREE_SIZE))
toolbox.decorate("mutate", dual_static_limit(MAX_TREE_DEPTH, MAX_TREE_SIZE))

def run_pytorch_gp():
    """
    Main GP evolution loop using PyTorch for neural network evaluation.
    
    Benefits of PyTorch version:
    1. GPU acceleration for fitness evaluation
    2. More efficient memory usage
    3. Better performance on large neural networks
    4. More flexible model architectures
    
    NEW: Enhanced with better bloat control and population monitoring
    """
    print("Starting PyTorch-accelerated Genetic Programming...")
    print(f"Using device: {device}")
    print(f"Population size: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print(f"Max tree size: {MAX_TREE_SIZE}, Max tree depth: {MAX_TREE_DEPTH}")
    
    # Initialize population
    print("\nGenerating initial population...")
    population = toolbox.population(n=POPULATION_SIZE)
    
    # NEW: Validate initial population
    print("\nValidating initial population...")
    for i, ind in enumerate(population):
        size = len(ind)
        height = ind.height
        valid = is_tree_valid(ind)
        print(f"Individual {i}: Size={size}, Height={height}, Valid={valid}")
        if not valid:
            print(f"WARNING: Invalid individual in initial population!")
    
    # Evaluate initial population
    print("\nEvaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    print("\nInitial population fitnesses:")
    for i, ind in enumerate(population):
        print(f"Individual {i}: Fitness={ind.fitness.values[0]:.6f}, Size={len(ind)}, Height={ind.height}")

    # Evolution loop
    for generation in range(GENERATIONS):
        print(f"\n=== Generation {generation + 1}/{GENERATIONS} ===")
        
        # Sort population by fitness (higher is better)
        population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        elite_pool_size = int(ELITISM_RATE * POPULATION_SIZE)
        elite_pool = population[:elite_pool_size]
        
        # NEW: Track population statistics
        sizes = [len(ind) for ind in population]
        heights = [ind.height for ind in population]
        print(f"Population stats - Size: avg={np.mean(sizes):.1f}, max={max(sizes)}, min={min(sizes)}")
        print(f"Population stats - Height: avg={np.mean(heights):.1f}, max={max(heights)}, min={min(heights)}")
        
        new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            operator_prob = random.random()

            if operator_prob < ELITISM_RATE:
                # Elitism: select from elite pool
                selected = toolbox.select(elite_pool)
                ind = selected[0]
                elite_copy = creator.Individual(ind)
                elite_copy.fitness.values = ind.fitness.values
                new_population.append(elite_copy)
                
            elif operator_prob < ELITISM_RATE + MUTATION_RATE:
                # Mutation
                selected = toolbox.select(population)
                ind = selected[0]
                mutant = toolbox.mutate(ind)[0]
                mutant.fitness.values = (toolbox.evaluate(mutant),)
                new_population.append(mutant)
                
            elif operator_prob < ELITISM_RATE + MUTATION_RATE + CROSSOVER_RATE:
                # Crossover
                parent1_selected = toolbox.select(population)
                parent2_selected = toolbox.select(population)
                parent1 = parent1_selected[0]
                parent2 = parent2_selected[0]
                
                offspring1, offspring2 = toolbox.mate(parent1, parent2)
                offspring1.fitness.values = (toolbox.evaluate(offspring1),)
                new_population.append(offspring1)
                
                if len(new_population) < POPULATION_SIZE:
                    offspring2.fitness.values = (toolbox.evaluate(offspring2),)
                    new_population.append(offspring2)
                    
            else:
                # Selection (copy)
                selected = toolbox.select(population)
                ind = selected[0]
                ind_copy = creator.Individual(ind)
                ind_copy.fitness.values = ind.fitness.values
                new_population.append(ind_copy)

        # Replace population
        population[:] = new_population
        
        # NEW: Validate new population
        invalid_count = 0
        for ind in population:
            if not is_tree_valid(ind):
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"WARNING: {invalid_count} invalid individuals in generation {generation + 1}")
        
        # Find and display best individual
        best_ind = max(population, key=lambda ind: ind.fitness.values[0])
        print(f"Generation {generation + 1}: Best Fitness = {best_ind.fitness.values[0]:.6f}")
        print(f"Best individual: Size={len(best_ind)}, Height={best_ind.height}")
        
        # Display best tree structure every few generations
        if generation % 2 == 0:
            print("\nBest tree structure:")
            print(str(best_ind)[:200] + "..." if len(str(best_ind)) > 200 else str(best_ind))

    # Final results
    print("\n=== Evolution Complete ===")
    final_best = max(population, key=lambda ind: ind.fitness.values[0])
    print(f"Final best fitness: {final_best.fitness.values[0]:.6f}")
    print(f"Final best tree size: {len(final_best)}")
    print(f"Final best tree height: {final_best.height}")
    print(f"Final best tree: {final_best}")
    
    # NEW: Final population statistics
    final_sizes = [len(ind) for ind in population]
    final_heights = [ind.height for ind in population]
    print(f"\nFinal population statistics:")
    print(f"Sizes - avg: {np.mean(final_sizes):.1f}, max: {max(final_sizes)}, min: {min(final_sizes)}")
    print(f"Heights - avg: {np.mean(final_heights):.1f}, max: {max(final_heights)}, min: {min(final_heights)}")
 
    nodes, edges, labels = graph(final_best)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

    return toolbox.compile(final_best)

"""
if __name__ == '__main__':
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Run the PyTorch-accelerated GP
    start = time.time()
    gp_crossover_func = run_pytorch_gp()
    
    print(gp_crossover_func)
    end = time.time()
    print(f"Total Time Taken : {(end-start):.2f} seconds")
    print("\nGP Evolution completed successfully! Crossover function is ready for GA.")
"""