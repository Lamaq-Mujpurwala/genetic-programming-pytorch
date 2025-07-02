from deap import base , creator , tools
import random 
creator.create("FitnessMax", base.Fitness , weights=(+1.0,) ) # Maximixation or Minimization is handled automatically
creator.create("Individual" , list ,fitness=creator.FitnessMax) # this creates an Individual type with Fitness Maximixation Objective


GENOME_SIZE = 20
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
GENERATIONS = 100

# Initialization
toolbox = base.Toolbox()
toolbox.register("attribute" , lambda: random.randint(0, 1)) # Since toolbox.reg expects only a func , we need to pass out a,b limit as a lambda func
toolbox.register("individual" , tools.initRepeat , creator.Individual , toolbox.attribute , n=GENOME_SIZE) # this is to create a single individual with the desiered genome length , the initRepeat repeats the attribute which is picking 0 or 1 for n times
toolbox.register("population" , tools.initRepeat , list ,toolbox.individual) # This creates the initial population , the number argument can be written directly inside here or at the time of calling it


# Operators

def calculate_fitness(individual):
    # return sum(individual) this is the original idea , but DEAP expects fitness to be a tuple of value
    return (sum(individual),)

def best_individual(population):
    best_idx = population.index(max(population , key=calculate_fitness))
    return best_idx


toolbox.register("mate" , tools.cxTwoPoint)
toolbox.register("mutate" , lambda ind: tools.mutFlipBit(ind, indpb=0.05)) #instead of mutate guass since that works with Float here we are dealing w binary genes
toolbox.register("select" , tools.selTournament , tournsize=5)
toolbox.register("evaluate",calculate_fitness)
toolbox.register("best_ind",best_individual)

def alogrithm():
    population = toolbox.population(n=100)

    fitness = map(toolbox.evaluate , population)
    for ind , fit in zip(population,fitness):
        ind.fitness.values = fit

    for generation in range(GENERATIONS):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        population[:] = offspring

        best_ind = max(population, key=lambda ind: ind.fitness.values)
        print(f"Generation {generation}: Best Fitness = {best_ind.fitness.values[0]}, Individual = {best_ind}")


if __name__ == '__main__':
    alogrithm()        
    
