from deap import creator , base , tools
import random
import array
import numpy

# Fitness - for Max problems se weights as 1.0 , for Min problem set as -1.0
creator.create("FitnessMin" , base.Fitness , weights=(1.0))

creator.create("FitnessMulti" , base.Fitness , weights=(-1.0 , 1.0)) #This creates a multi fitness func that minimizes the first objective and maximixes the second one

# Types of Individuals
toolbox = base.Toolbox()
GENOME_LEN = IND_SIZE = 10
## Indiviudal
creator.create("Individual" , list , fitness = creator.FitnessMin) #or multi . max whatever

## 1. List of Floats
toolbox.register("attr_float", random.random)
toolbox.register("individual", # alias - name it is called as
                tools.initRepeat, # How to fill the list
                creator.Individual, # Type of every object in the list
                toolbox.attr_float, #DataType of every object in the list
                n=IND_SIZE # Genes length
                )

## 2. Evolution Strategy - It has two lists , Individual list with the genes and the Startegy list with mutation parameters

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d",
               fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

def initES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

IND_SIZE = 10
MIN_VALUE, MAX_VALUE = -5., 5.
MIN_STRAT, MAX_STRAT = -1., 1. 

toolbox = base.Toolbox()
toolbox.register("individual", initES, creator.Individual,
                 creator.Strategy, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRAT, 
                 MAX_STRAT)

if __name__ == '__main__':
    output = toolbox.individual()
    print(output)