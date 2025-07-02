#define _WIN32_WINNT 0x0600  // Enable BCryptGenRandom support
#define STRING_SIZE 100
#define POPULATION_SIZE 20

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <bcrypt.h>
#include <string.h>

#pragma comment(lib, "bcrypt.lib")

typedef struct{
    char genes[STRING_SIZE];
    int fitness;
} Individual;

static char *genes = "abcdefghijklmnopqrstuvwxyz 1234567890";
static char *target = "hello world 123";

int random_number(int min, int max){
    int range = (max - min) + 1;
    unsigned int val;

    if (BCryptGenRandom(NULL, (PUCHAR)&val, sizeof(val), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        fprintf(stderr, "Failed to generate random number.\n");
        exit(1); // exit the program to avoid using uninitialized val
    }

    return min + (val % range);
}

char mutated_genes(){
    int len = strlen(genes);
    int r = random_number(0, len - 1);
    return genes[r];
}

void create_sequence(char *gnome) {
    int seq = strlen(target);
    
    for (int i = 0; i < seq; i++) {
        gnome[i] = mutated_genes(); 
    }
    gnome[seq] = '\0';
}



int fitness(char *chromosomes){
    int len = strlen(chromosomes);
    int i;
    int score=0;
    for(i=0;i<len;i++){
        if(chromosomes[i]==target[i]){
            score++;
        }
    }
    return score;
}

Individual mate(Individual *parent1 , Individual *parent2 ){
    Individual child;
    int len = strlen(target);
    for (int i = 0; i < len; i++) {
        float prob = random_number(0, 100) / 100.0;
        if (prob < 0.45)
            child.genes[i] = parent1->genes[i];
        else if (prob < 0.90)
            child.genes[i] = parent2->genes[i];
        else
            child.genes[i] = mutated_genes();
    }
    child.genes[len] = '\0';
    child.fitness = fitness(child.genes);
    return child;
    
}



void initialize_population(Individual population[] , int size){
    int i;
    for(i=0;i<size;i++){
        create_sequence(population[i].genes);
        population[i].fitness = fitness(population[i].genes);
    }
}

int compare_individuals(const void *a, const void *b) {
    const Individual *indA = (const Individual *)a;
    const Individual *indB = (const Individual *)b;

    // For descending order
    return indB->fitness - indA->fitness;
}





int main() {
    //char gene[] = "hekko gorlt 453";
    //printf("%d\n", fitness(gene));
    //int num = random_number(0,100);
    //float p = num/100.0;
    //printf("%d\n",num);
    //printf("%f\n",p);

    //initialize_population(population,POPULATION_SIZE); **TO CHECK IF POPULATION WORKS**
    //int i;
    /*for(i=0;i<20;i++){
        printf("Individual %d: \n",(i+1));
        printf("\t Genes: %s",population[i].genes);
        printf("\t Fitness: %d\n",population[i].fitness);
        printf("\n");
    }*/ 

    // ACTUAL DRIVER CODE STARTS HERE
    Individual population[POPULATION_SIZE];
    initialize_population(population,POPULATION_SIZE); // This is the first (here 0) generation of the population which already has genes and some fitnesss vlaue
    int generation=0;
    int target_fitness = fitness(target); //Defining it here so dont have to make fucntion calls in while loop
    int found=0;

    while(found==0){

        qsort(population,POPULATION_SIZE,sizeof(Individual),compare_individuals);

        if(population[0].fitness == target_fitness){
            found=1;
            printf("Target fitness reached in generation %d\n",generation);
            break;
        }

        // New generation
        Individual new_generation[POPULATION_SIZE]; // this is okay for 100 ish population size as stack memory will not overflow , but for larger data we might have to swtich with heap

        // Elietism - 10% of fittest go to next generation
        int s = (10*POPULATION_SIZE)/100;
        int i;
        for(i=0;i<s;i++){
            new_generation[i] = population[i];
        }
        // Single Parent Crossover 
         // For the rest 90% new generation , the top 50% fittest will mate to produce new offspring
        for(i;i<POPULATION_SIZE;i++){
            int r = random_number(0, POPULATION_SIZE / 2 - 1);  // top 50%
            Individual parent1 = population[r];
            r = random_number(0, POPULATION_SIZE / 2 - 1);
            Individual parent2 = population[r];
            Individual offspring =  mate(&parent1,&parent2);
            new_generation[i] = offspring;
        }

        // population = new_generation; - C does not allow copying statically declared arrays , so we have to manually copy the new_generation to the population;
        /* One way we can do it is m=by memcpy which looks cleaner:
            memcpy(population, new_generation, sizeof(Individual) * POPULATION_SIZE);
           But since it is simpler to understand whats happening with a loop so i'll implement a loop
        */

        for (int i = 0; i < POPULATION_SIZE; i++) {
            population[i] = new_generation[i];
        }

        //Printing the generation results

        printf("Generation %d\t",generation);
        printf("String %s \t",population[0].genes);
        printf("Fitness %d \n",population[0].fitness);

        //Moving ahead with generation
        generation++;

    }

    printf("Generation %d\t",generation);
    printf("String %s \t",population[0].genes);
    printf("Fitness %d \n",population[0].fitness);

    return 0;
}
