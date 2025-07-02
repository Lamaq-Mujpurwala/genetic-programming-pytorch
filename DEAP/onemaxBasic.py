import random

MAX_GENERATIONS = 100
POPULATION_SIZE = 200
GENOME_LENGTH = 50
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7

def create_individual(genome_length):
    return [random.randint(0, 1) for _ in range(genome_length)]

def init_population(population_size, genome_length):
    return [create_individual(genome_length) for _ in range(population_size)]

def fitness_function(individual):
    return sum(individual)

def select_parents(population):
    return random.choice(population)

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1[:], parent2[:]

def mutate(genome):
    return [gene if random.random() > MUTATION_RATE else 1 - gene for gene in genome]

def genetic_algorithm():
    population = init_population(POPULATION_SIZE, GENOME_LENGTH)

    for generation in range(MAX_GENERATIONS):
        population.sort(key=fitness_function, reverse=True)
        fitness_scores = [fitness_function(ind) for ind in population]
        best_fitness = fitness_scores[0]
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Elitism - top 10%
        elite_count = int(0.10 * POPULATION_SIZE)
        elites = population[:elite_count]
        new_population = elites.copy()

        # Generate rest of the population
        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parents(population)
            parent2 = select_parents(population)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutate(offspring1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(offspring2))

        population = new_population

    best_index = fitness_scores.index(max(fitness_scores))
    best_individual = population[best_index]
    print(f"\nBest Solution = {best_individual}")
    print(f"Fitness = {max(fitness_scores)}")

if __name__ == '__main__':
    genetic_algorithm()
