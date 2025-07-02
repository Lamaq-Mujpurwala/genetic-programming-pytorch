import torch
from deap import tools, creator, base
import numpy as np
import pandas as pd
import random
import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up to the root of the project
sys.path.append(str(project_root))

# Import our new helper functions
from utils.pytorch.ga_helpers import (
    get_weights_pytorch,
    reassemble_model_pytorch,
    get_fitness_pytorch,
    mutation_pytorch,
    check_device
)

from datasets import DatasetFactory, DataPreprocessor

# GA Parameters
POPULATION_SIZE = 10
GENERATIONS = 40
ELITISM_RATE = 0.1
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.4  # Not used in this version

def setup_ga(dataset_name: str = "wine"):
    """
    Setup the genetic algorithm with the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to use (default: "wine")
        
    Returns:
        tuple: (toolbox, X_train_scaled, X_test_scaled, y_train, y_test)
    """
    # Initialize dataset and preprocessor
    factory = DatasetFactory()
    dataset = factory.get_dataset(dataset_name)
    preprocessor = DataPreprocessor()
    
    # Get and preprocess the data
    data = dataset.get_data()
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.prepare_data(
        data,
        target_column=dataset.get_target_name()
    )
    
    # Convert pandas Series to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    print(f"Dataset loaded - Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    
    # Get dataset information for model creation
    dataset_info = dataset.get_dataset_info()
    input_size = len(dataset_info['feature_names'])
    num_classes = len(np.unique(y_train))
    
    # Setup device
    device = check_device()
    
    # DEAP Setup - Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize loss
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    
    def pytorch_mutation_wrapper(offspring):
        """Wrapper function for PyTorch mutation to maintain DEAP compatibility."""
        return mutation_pytorch(
            offspring, 
            X_train_scaled, 
            y_train, 
            dataset_name=dataset_name,
            input_size=input_size,
            num_classes=num_classes,
            device=device
        )
    
    def pytorch_fitness_wrapper(individual):
        """Wrapper function for PyTorch fitness evaluation to maintain DEAP compatibility."""
        return get_fitness_pytorch(
            individual, 
            X_train_scaled, 
            X_test_scaled, 
            y_train, 
            y_test,
            dataset_name=dataset_name,
            input_size=input_size,
            num_classes=num_classes,
            device=device
        )
    
    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", 
        tools.initRepeat, 
        creator.Individual, 
        lambda: get_weights_pytorch(dataset_name, input_size, num_classes), 
        n=1
    )
    toolbox.register("populate", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", pytorch_fitness_wrapper)
    toolbox.register("select", tools.selTournament, k=1, tournsize=2, fit_attr='fitness')
    toolbox.register("mutate", pytorch_mutation_wrapper)
    
    return toolbox, X_train_scaled, X_test_scaled, y_train, y_test

def run_genetic_algorithm(dataset_name: str = "wine"):
    """
    Main function to run the genetic algorithm with PyTorch neural networks.
    
    Args:
        dataset_name (str): Name of the dataset to use (default: "wine")
        
    Returns:
        pd.DataFrame: Dataset suitable for GP training
    """
    # Setup GA with the specified dataset
    toolbox, X_train_scaled, X_test_scaled, y_train, y_test = setup_ga(dataset_name)
    
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
        elite_pool_size = max(1, int(ELITISM_RATE * POPULATION_SIZE))
        elite_pool = population[:elite_pool_size]
        
        print(f"Elite pool size: {elite_pool_size}")
        print(f"Current best loss: {population[0].fitness.values[0]:.4f}")
        
        # Create new generation using probabilistic operator selection
        new_population = []
        operator_counts = {'elitism': 0, 'mutation': 0, 'reproduction': 0}
        
        while len(new_population) < POPULATION_SIZE:
            operator_prob = random.random()
            
            if operator_prob < ELITISM_RATE:
                # Elitism: Select from elite pool
                selected = toolbox.select(elite_pool)
                ind = selected[0]
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
                    ind.fitness.values = (10.0,)
        
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
    
    # Create dataset for GP training
    if len(population) % 2 != 0:
        population = population[:-1]
    
    data = []
    for i in range(0, len(population), 2):
        parent1 = population[i]
        p1_fitness = parent1.fitness.values[0]
        parent2 = population[i+1]
        p2_fitness = parent2.fitness.values[0]
        
        data.append([
            {'parent1': parent1, 'fitness': p1_fitness},
            {'parent2': parent2, 'fitness': p2_fitness}
        ])
    
    dataset = pd.DataFrame(data=data, columns=['parent1', 'parent2'])
    print(f"\nDataset for GP successfully created with {len(dataset)} pairs")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    return dataset

if __name__ == '__main__':
    try:
        print("Starting PyTorch-based Genetic Algorithm for Neural Network Evolution")
        print("=" * 70)
        
        # Run the genetic algorithm with the wine dataset
        dataset = run_genetic_algorithm("wine")
        
        print("Evolution completed successfully!")
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred during evolution: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 