import numpy as np
import operator as op
import random
import functools
import torch
import pandas as pd
from deap import base, creator, gp, tools
from deap.gp import PrimitiveSetTyped, PrimitiveTree, genHalfAndHalf, graph, cxOnePoint, mutUniform, staticLimit , compile
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pathlib import Path
import sys

# Import PyTorch model functions
from utils.gp_helpers import get_ga_fitness, get_model_weights, get_fitness
from utils.pytorch.models import get_model_architecture
from datasets.wine_dataset import WineDataset
from datasets.preprocessing import DataPreprocessor

# Import GA function to get dataset
from ga_without_crossover import run_genetic_algorithm

def is_tree_valid(tree, max_size, max_depth):
    """Check if a tree meets size and depth constraints."""
    try:
        return len(tree) <= max_size and tree.height <= max_depth
    except (IndexError, AttributeError):
        return False

def generate_constrained_tree(pset, min_depth, max_depth, max_size, max_attempts=100):
    """Generate a single tree that satisfies both depth and size constraints."""
    for attempt in range(max_attempts):
        current_max_depth = min(min_depth + (attempt // 10), max_depth)
        try:
            tree = genHalfAndHalf(pset, min_depth, current_max_depth)
            individual = creator.Individual(tree)
            if is_tree_valid(individual, max_size, max_depth):
                return individual
        except Exception:
            continue
    # Fallback: minimal tree
    return create_minimal_tree(pset)

def create_minimal_tree(pset):
    from deap.gp import Terminal, Primitive
    tree = creator.Individual([
        Primitive('Sum', [np.ndarray, np.ndarray], np.ndarray),
        Terminal('P1', True, np.ndarray),
        Terminal('P2', True, np.ndarray)
    ])
    return tree

def generate_bloatless_population(pset, n, min_depth, max_depth, max_size):
    population = []
    failed_attempts = 0
    max_failed_attempts = n * 10
    while len(population) < n and failed_attempts < max_failed_attempts:
        try:
            ind = generate_constrained_tree(pset, min_depth, max_depth, max_size)
            if is_tree_valid(ind, max_size, max_depth):
                population.append(ind)
            else:
                failed_attempts += 1
        except Exception:
            failed_attempts += 1
    while len(population) < n:
        population.append(create_minimal_tree(pset))
    return population

def vector_sum(parent1, parent2):
    return parent1 + parent2

def vector_mean(parent1, parent2):
    return (parent1 + parent2) / 2

def one_point_crossover(i_point, w1, w2):
    i_point = int(i_point)
    return np.concatenate([w1[:i_point], w2[i_point:]])

def compare_mean_less(w1, w2):
    return np.mean(w1) < np.mean(w2)

def compare_norm_greater(w1, w2):
    return np.linalg.norm(w1) > np.linalg.norm(w2)

def select_fitness(condition, F1, F2):
    return F1 if condition else F2

def select_weights(condition, W1, W2):
    return W1 if condition else W2

def get_random_crossover_point(parent_length):
    return random.randint(1, parent_length - 1)

def get_gp_fitness(individual, dataset, toolbox, reassemble_model_func, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device):
    func = toolbox.compile(expr=individual)
    fitness = 0
    for i, row in dataset.iterrows():
        parent1 = row['parent1']['parent1'][0]
        p1_fitness = row['parent1']['fitness']
        parent2 = row['parent2']['parent2'][0]
        p2_fitness = row['parent2']['fitness']
        try:
            model = reassemble_model_func(np.concatenate([parent1, parent2, p1_fitness, p2_fitness]))
            model.to(device)
            model.eval()
            with torch.no_grad():
                X_train_tensor_batch = X_train_tensor[i].unsqueeze(0)
                X_test_tensor_batch = X_test_tensor[i].unsqueeze(0)
                y_train_tensor_batch = y_train_tensor[i].unsqueeze(0)
                y_test_tensor_batch = y_test_tensor[i].unsqueeze(0)
                output_train = model(X_train_tensor_batch)
                output_test = model(X_test_tensor_batch)
                train_loss = get_ga_fitness(output_train.cpu().numpy())
                test_loss = get_ga_fitness(output_test.cpu().numpy())
                fitness += float(train_loss / test_loss)
        except Exception as e:
            print(f"Error evaluating individual on datapoint {i}: {e}")
            return float('-inf')
    return fitness / len(dataset)

def safe_mate(ind1, ind2, max_size, max_depth):
    for _ in range(5):
        try:
            parent1_copy = creator.Individual(ind1)
            parent2_copy = creator.Individual(ind2)
            offspring1, offspring2 = cxOnePoint(parent1_copy, parent2_copy)
            if (is_tree_valid(offspring1, max_size, max_depth) and is_tree_valid(offspring2, max_size, max_depth)):
                return offspring1, offspring2
        except Exception:
            continue
    return ind1, ind2

def safe_mutate(ind, pset, min_depth, max_depth, max_size):
    for _ in range(5):
        try:
            ind_copy = creator.Individual(ind)
            mutant = mutUniform(ind_copy, expr=lambda pset=pset: generate_constrained_tree(pset, min_depth, max_depth, max_size), pset=pset)[0]
            if is_tree_valid(mutant, max_size, max_depth):
                return mutant,
        except Exception:
            continue
    return ind,

def main():
    # Parameters
    MIN_TREE_DEPTH = 2
    MAX_TREE_DEPTH = 4
    MAX_TREE_SIZE = 20
    POPULATION_SIZE = 15
    GENERATIONS = 10
    ELITISM_RATE = 0.1
    MUTATION_RATE = 0.3
    CROSSOVER_RATE = 0.6
    DATASET_NAME = "wine"

    # --- Prepare data and model ---
    dataset_handler = WineDataset()
    data = dataset_handler.get_data()
    preprocessor = DataPreprocessor()
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.prepare_data(
        data, target_column=dataset_handler.get_target_name())
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))
    # Model class for reassembly
    model_class = functools.partial(get_model_architecture, DATASET_NAME, input_size, num_classes)
    # Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    # Reassembly function
    def reassemble_model_func(flat_vector):
        return get_model_architecture(DATASET_NAME, input_size, num_classes)
    # ---
    # Get dataset from GA
    ga_dataset = run_genetic_algorithm()
    parent_length = len(ga_dataset.iloc[0]['parent1']['parent1'][0])

    # Primitive set
    pset = PrimitiveSetTyped("main", [np.ndarray, np.ndarray, float, float], np.ndarray)
    pset.addTerminal(1, float)
    pset.addTerminal(2, float)
    pset.addTerminal(0.5, float)
    pset.addTerminal(1, bool)
    pset.addTerminal(0, bool)
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
    pset.addEphemeralConstant("fixed_crossover_point", functools.partial(get_random_crossover_point, parent_length), int)
    pset.addPrimitive(one_point_crossover, [int, np.ndarray, np.ndarray], np.ndarray, name="OnePointCrossover")
    pset.renameArguments(ARG0="P1", ARG1="P2", ARG2="F1", ARG3="F2")

    # DEAP types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", lambda: generate_constrained_tree(pset, MIN_TREE_DEPTH, MAX_TREE_DEPTH, MAX_TREE_SIZE))
    toolbox.register("individual", lambda: toolbox.expr())
    toolbox.register("population", lambda n=POPULATION_SIZE: generate_bloatless_population(pset, n, MIN_TREE_DEPTH, MAX_TREE_DEPTH, MAX_TREE_SIZE))
    toolbox.register("compile", compile, pset=pset)
    toolbox.register(
        "evaluate",
        lambda ind: get_gp_fitness(
            ind, ga_dataset, toolbox,
            lambda flat_vector: reassemble_model_func(flat_vector),
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device
        )
    )
    toolbox.register("select", tools.selTournament, k=1, tournsize=3, fit_attr='fitness')
    toolbox.register("mate", lambda ind1, ind2: safe_mate(ind1, ind2, MAX_TREE_SIZE, MAX_TREE_DEPTH))
    toolbox.register("mutate", lambda ind: safe_mutate(ind, pset, MIN_TREE_DEPTH, MAX_TREE_DEPTH, MAX_TREE_SIZE))

    # Initial population
    population = toolbox.population()

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    for generation in range(GENERATIONS):
        print(f"\n=== Generation {generation + 1}/{GENERATIONS} ===")
        population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        elite_pool_size = int(ELITISM_RATE * POPULATION_SIZE)
        elite_pool = population[:elite_pool_size]
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            operator_prob = random.random()
            if operator_prob < ELITISM_RATE:
                selected = toolbox.select(elite_pool)
                ind = selected[0]
                elite_copy = creator.Individual(ind)
                elite_copy.fitness.values = ind.fitness.values
                new_population.append(elite_copy)
            elif operator_prob < ELITISM_RATE + MUTATION_RATE:
                selected = toolbox.select(population)
                ind = selected[0]
                mutant = toolbox.mutate(ind)[0]
                mutant.fitness.values = (toolbox.evaluate(mutant),)
                new_population.append(mutant)
            elif operator_prob < ELITISM_RATE + MUTATION_RATE + CROSSOVER_RATE:
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
                selected = toolbox.select(population)
                ind = selected[0]
                ind_copy = creator.Individual(ind)
                ind_copy.fitness.values = ind.fitness.values
                new_population.append(ind_copy)
        population[:] = new_population
        best = max(population, key=lambda ind: ind.fitness.values[0])
        print(f"Generation {generation + 1}: Best fitness = {best.fitness.values[0]:.6f}")
    best = max(population, key=lambda ind: ind.fitness.values[0])
    print(f"Best individual: {best}")
    # Visualization
    nodes, edges, labels = graph(best)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

if __name__ == "__main__":
    main() 