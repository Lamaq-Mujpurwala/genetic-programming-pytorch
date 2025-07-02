import torch
from deap import tools , creator , base
import numpy as np
import pandas as pd
import random

import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GP_for_RGECO import run_pytorch_gp


from PyTorch.GApytorch import (
    get_weights_pytorch, 
    reassemble_model_pytorch, 
    get_fitness_pytorch, 
    mutation_pytorch,
    check_device
)
device = check_device()

gp_crossover_function = run_pytorch_gp() #this returns a lambda func

from Datasets.get_dataset import get_scaled_data
from Datasets.wine_dataset import get_wine_data
dataset = get_wine_data()
X_train_scaled, X_test_scaled, y_train, y_test = get_scaled_data(dataset=dataset, target_column='target')

# Convert pandas Series to numpy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print(f"Dataset loaded - Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

"""
GA parameters from the paper are:

    Parameter               Value
    Population size         10
    Mutation rate           0.5
    Crossover rate          0.4
    Elitism rate            0.1
    Tournament size         2
"""


"""
Generations per dataset : 

    Dataset             Number of generations
    Thumbnail               40
    CIFAR-10                45
    Higgs                   50
    Bean                    75
    Wine                    40

"""


POPULATION_SIZE = 10  
GENERATIONS = 40
ELITISM_RATE = 0.1
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.4 # -> Here the crossover rates matter

# DEAP Setup - Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize loss
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

def pytorch_mutation_wrapper(offspring):
    """
    Wrapper function for PyTorch mutation to maintain DEAP compatibility.
    
    Args:
        offspring: Individual to mutate
    
    Returns:
        mutated_individual: Individual after mutation (brief training)
    """
    return mutation_pytorch(offspring, X_train_scaled, y_train, device)

def pytorch_fitness_wrapper(individual):
    """
    Wrapper function for PyTorch fitness evaluation to maintain DEAP compatibility.
    
    Args:
        individual: Individual to evaluate
    
    Returns:
        fitness_tuple: Tuple containing loss value for DEAP
    """
    return get_fitness_pytorch(individual, X_train_scaled, X_test_scaled, y_train, y_test, device)


toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual, get_weights_pytorch, n=1)
toolbox.register("populate", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", pytorch_fitness_wrapper)
toolbox.register("select", tools.selTournament, k=1, tournsize=2, fit_attr='fitness')
toolbox.register("crossover", gp_crossover_function)
toolbox.register("mutate", pytorch_mutation_wrapper) 


def run_genetic_algorithm():
    """
    Main function to run the genetic algorithm with PyTorch neural networks.
    Returns a dataset suitable for GP training.
    """
    print("Initializing population...")
    population = toolbox.populate(POPULATION_SIZE)
    
    print("Evaluating initial population...")
    # Evaluate the entire initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Print initial population statistics
    initial_losses = [ind.fitness.values[0] for ind in population]
    print(f"Initial population - Best loss: {min(initial_losses):.4f}, "
          f"Worst loss: {max(initial_losses):.4f}, "
          f"Average loss: {np.mean(initial_losses):.4f}")
    
    # Evolution loop
    print(f"\nStarting evolution for {GENERATIONS} generations...")
    
    for generation in range(GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{GENERATIONS} ---")
        
        # Sort population by fitness (lower loss = better fitness)
        population.sort(key=lambda ind: ind.fitness.values[0])
        
        # Create elite pool
        elite_pool_size = max(1, int(ELITISM_RATE * POPULATION_SIZE))  # Ensure at least 1 elite
        elite_pool = population[:elite_pool_size]
        
        print(f"Elite pool size: {elite_pool_size}")
        print(f"Current best loss: {population[0].fitness.values[0]:.4f}")
        
        # Create new generation using probabilistic operator selection
        new_population = []
        operator_counts = {'elitism': 0, 'mutation': 0, 'crossover': 0, 'reproduction': 0}
        
        while len(new_population) < POPULATION_SIZE:
            operator_prob = random.random()
            
            if operator_prob < ELITISM_RATE:
                # Elitism: Select from elite pool
                selected = toolbox.select(elite_pool)
                ind = selected[0]
                # Create proper copy of elite individual
                elite_copy = creator.Individual([ind[0].copy()])
                elite_copy.fitness.values = ind.fitness.values
                new_population.append(elite_copy)
                operator_counts['elitism'] += 1
                
            elif operator_prob < ELITISM_RATE + MUTATION_RATE:
                # Mutation: Select individual and mutate
                selected = toolbox.select(population)
                ind = selected[0]
                try:
                    mutant = toolbox.mutate(ind)
                    new_population.append(mutant)
                    operator_counts['mutation'] += 1
                except Exception as e:
                    print(f"Mutation failed: {e}")
                    # Fallback to reproduction
                    ind_copy = creator.Individual([ind[0].copy()])
                    ind_copy.fitness.values = ind.fitness.values
                    new_population.append(ind_copy)
                    operator_counts['reproduction'] += 1
            elif operator_prob < ELITISM_RATE + MUTATION_RATE + CROSSOVER_RATE:
                # Crossover: Select two parents and create offspring
                parent1_selected = toolbox.select(population)
                parent2_selected = toolbox.select(population)
                parent1 = parent1_selected[0]
                parent1_fitness = parent1.fitness.values[0]
                parent2 = parent2_selected[0]
                parent2_fitness = parent2.fitness.values[0]
                
                try:
                    offspring = toolbox.crossover(parent1, parent2, parent1_fitness, parent2_fitness)
                    print(f"\nCrossover successful - Offspring type: {type(offspring)}, shape: {offspring.shape if hasattr(offspring, 'shape') else 'No shape'}")
                    
                    # Create new individual and add to population
                    offspring_ind = creator.Individual(offspring)  # Wrap in list like other operations
                    
                    # Evaluate offspring fitness
                    offspring_fitness = toolbox.evaluate(offspring_ind)
                    offspring_ind.fitness.values = offspring_fitness
                    
                    # Analyze crossover effectiveness
                    parent_avg_fitness = (parent1_fitness + parent2_fitness) / 2
                    improvement = parent_avg_fitness - offspring_fitness[0]  # Positive means improvement
                    print(f"Parent avg fitness: {parent_avg_fitness:.4f}")
                    print(f"Offspring fitness: {offspring_fitness[0]:.4f}")
                    print(f"Improvement: {improvement:.4f} ({'Better' if improvement > 0 else 'Worse'})")
                    
                    new_population.append(offspring_ind)
                    operator_counts['crossover'] += 1
                    
                except Exception as e:
                    print(f"\nCrossover failed with error: {str(e)}")
                    # Fallback to reproduction of better parent
                    better_parent = parent1 if parent1_fitness < parent2_fitness else parent2
                    ind_copy = creator.Individual([better_parent[0].copy()])
                    ind_copy.fitness.values = better_parent.fitness.values
                    new_population.append(ind_copy)
                    operator_counts['reproduction'] += 1
            else:
                # Reproduction: Copy individual without modification
                selected = toolbox.select(population)
                ind = selected[0]
                ind_copy = creator.Individual([ind[0].copy()])
                ind_copy.fitness.values = ind.fitness.values
                new_population.append(ind_copy)
                operator_counts['reproduction'] += 1
        
        print(f"Operator usage - Elitism: {operator_counts['elitism']}, "
              f"Mutation: {operator_counts['mutation']}, "
              f"Crossover: {operator_counts['crossover']}, "
              f"Reproduction: {operator_counts['reproduction']}")
        
        # Replace old population with new population
        population[:] = new_population
        
        # Evaluate individuals that don't have valid fitness
        individuals_to_evaluate = [ind for ind in population if not ind.fitness.valid]
        
        if individuals_to_evaluate:
            print(f"Evaluating {len(individuals_to_evaluate)} new individuals...")
            try:
                fitnesses = list(map(toolbox.evaluate, individuals_to_evaluate))
                for ind, fit in zip(individuals_to_evaluate, fitnesses):
                    ind.fitness.values = fit
            except Exception as e:
                print(f"Evaluation failed: {e}")
                # Assign high loss to failed individuals
                for ind in individuals_to_evaluate:
                    ind.fitness.values = (10.0,)  # High loss value
        
        # Print generation statistics
        current_losses = [ind.fitness.values[0] for ind in population]
        best_loss = min(current_losses)
        worst_loss = max(current_losses)
        avg_loss = np.mean(current_losses)
        
        print(f"Generation {generation + 1} complete - "
              f"Best: {best_loss:.4f}, Worst: {worst_loss:.4f}, Average: {avg_loss:.4f}")
        
        # Optional: Early stopping if loss is very low
        if best_loss < 0.01:
            print(f"Early stopping: Loss below threshold at generation {generation + 1}")
            break
    
    # Final results
    population.sort(key=lambda ind: ind.fitness.values[0])
    best_individual = population[0]
    
    print(f"\n=== EVOLUTION COMPLETE ===")
    print(f"Best individual loss: {best_individual.fitness.values[0]:.4f}")
    """
    # Ensure even number of individuals for pairing
    if len(population) % 2 != 0:
        population = population[:-1]  # Remove last individual if odd number
    
    data = []
    for i in range(0, len(population), 2):
        parent1 = population[i]
        p1_fitness = parent1.fitness.values[0]
        parent2 = population[i+1]
        p2_fitness = parent2.fitness.values[0]
        
        # Debug prints for types
        print(f"Type of parent1: {type(parent1)}")
        print(f"Type of parent1[0]: {type(parent1[0])}")
        print(f"Type of p1_fitness: {type(p1_fitness)}")
        print(f"Shape of parent1[0]: {parent1[0].shape}")
        
        data.append([
            {'parent1': parent1, 'fitness': p1_fitness}, 
            {'parent2': parent2, 'fitness': p2_fitness}
        ])

    dataset = pd.DataFrame(data=data, columns=['parent1', 'parent2'])
    print("\nDataset structure:")
    print(f"Type of dataset: {type(dataset)}")
    print(f"Type of first cell: {type(dataset.iloc[0,0])}")
    print(f"Keys in first cell: {dataset.iloc[0,0].keys()}")
    print(dataset)
    print(f"Dataset for GP successfully created with {len(dataset)} pairs")
    """
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

    return dataset


run_genetic_algorithm()

'''
    print(f"Best individual loss: {best_individual.fitness.values[0]:.4f}")
    
    # Optional: Test the best individual
    print("\nTesting best individual...")
    try:
        test_model = reassemble_model_pytorch(best_individual, device)
        print("Best model successfully created and ready for deployment!")
    except Exception as e:
        print(f"Error creating best model: {e}")
    
    return population, best_individual

if __name__ == '__main__':
    try:
        print("Starting PyTorch-based Genetic Algorithm for Neural Network Evolution")
        print("=" * 70)
        
        # Run the genetic algorithm
        final_population, best_individual = run_genetic_algorithm()
        
        print(f"Best final loss: {best_individual.fitness.values[0]:.4f}")
        print(f"Population : {final_population}")
        print(f"Population Individual : {final_population[0]}")

        i=0
        data = []
        while i < POPULATION_SIZE:
            parent1 = final_population[i]
            p1_fitness = final_population[i].fitness.values[0]
            parent2 = final_population[i+1]
            p2_fitness = final_population[i+1].fitness.values[0]
            data.append([{'parent1': parent1 , 'fitness' : p1_fitness}, {'parent2': parent2 , 'fitness' : p2_fitness} ])
            i+=2

        # GPU memory cleanup if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
        
        print("Evolution completed successfully!")
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred during evolution: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
'''
