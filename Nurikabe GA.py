# http://www.logicgamesonline.com/nurikabe/
# 5x5 Nurikabe Genetic Algorithm
# Spencer Young
# Stanley Do

# Imports
import random
from operator import itemgetter
import time
from numpy.random import choice
import itertools
import math
import sys

# SAMPLE
# 0 0 0 5 0
# 0 0 0 0 0
# 0 1 0 2 0
# 0 0 0 0 0
# 0 6 0 0 0

# SOLUTION
# 1 1 1 5 1
# 0 0 0 0 0
# 0 1 0 2 1
# 1 0 0 0 0
# 1 6 1 1 1

best_individual =[(0,3),(0,0),(0,1),(0,2),(0,4),(2,1),(2,3),(2,4),(4,1),(3,0),(4,0),(4,2),(4,3),(4,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,2),(3,1),(3,2),(3,3),(3,4)]

# Change breed to not maintain the top X elites, rather use a probability function to figure that out.
# Crossover should move an island from p1 into the child, then the rest into c1. Need to handle the duplicates.

# The class that combines everything to be called in main

# CONSTANTS
# Specify the grid size
# \/ \/ \/ \/ \/
# grid_size = 5
grid_size = 6
# grid_size = 7
# grid_size = 10
# /\ /\ /\ /\ /\

list_size = grid_size * grid_size

# The main island coordinates (x,y): value
# GRID SIZE 5
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(0, 3): 5, (2, 1): 1, (2, 3): 2, (4, 1): 6}
# center_coords = {(0,0):1, (2,0):7, (3,3):1}
# center_coords = {(1,4):4, (3,1):1, (3,3):1}
# center_coords = {(0,0):5, (0,2):1, (0,4):3, (4,0):1, (4,2):1, (4,4):1}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 6
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
center_coords = {(1,1):1, (2,0):5, (2,2):3, (4,2):2, (4,5):6}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 7
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(1,2): 3, (1,4): 4, (2,1): 1, (2,5): 1, (3,3): 1, (4,1): 4, (4,5): 1, (5,2): 1, (5,4): 4}
# center_coords = {(0,1):6, (0,3):2, (2,6):5, (3,5):6, (5,5):1}
# center_coords = {(0,0):19}
# center_coords = {(0,0):5, (0,4):1, (0,6):7, (2,0):4, (4,6):1, (6,0):5, (6,2):3, (6,6):1}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 10
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(9,9):1, (1,4):1, (1,6):2, (3,1): 1, (3,4):6, (4,6):1, (4,8):4, (5,1):2, (5,3):2, (6,5):2, (6,8):2, (8,3):3, (8,5):3, (9,9):2}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

center_coords_keys = list(center_coords.keys())
center_coords_vals = list(center_coords.values())
max_swaps = max(center_coords_vals) - 1

# Cumulative sum for indexing within an individual
# Every number but the last indicates where an island starts.
# The last number indicates where the ocean starts
s = 0
cum_sum = [0]
for x in center_coords_vals:
    s = s + x
    cum_sum.append(s)

# All the indices of where islands start.
cum_sum_butlast = cum_sum[:-1]
cum_sum_withmax = cum_sum.copy()
cum_sum_withmax.append(list_size)

max_islands = s
max_waters = list_size - s

# A list that contains all of the (x,y) coordinates corresponding to the grid size.
all_coords = [(x, y) for y in range(grid_size) for x in range(grid_size)]

# A list that does not contain the center coords.
valid_coords = [x for x in all_coords if x not in center_coords]

# Dictionary which contains all of the valid edges for the corresponding coordinate
adjacencies = dict()
for coord in all_coords:
    adjacents = []
    x, y = coord
    adjacents += [(x+1, y)] if x+1 < grid_size else []
    adjacents += [(x, y+1)] if y+1 < grid_size else []
    adjacents += [(x-1, y)] if x-1 >= 0 else []
    adjacents += [(x, y-1)] if y-1 >= 0 else []
    adjacencies[coord] = adjacents

# TODOS .... Need to add multiple Populations and find a way to do multi-objective fitness.
class NurikabeGA():

    def __init__(self, grid_size, center_coords, generations, print_interval):
        # Grid size indicates a NxN grid
        self.grid_size = grid_size

        # Specifies the center island coordinates
        self.center_coords = center_coords

        # Creates a list of all possible coordinates in a 5x5 grid
        self.gene_pool = [(x, y) for y in range(self.grid_size)
                          for x in range(self.grid_size) if (x, y) not in self.center_coords]

        self.generations = generations

        self.print_interval = print_interval

    def geneticAlgorithm(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False):
        startTime = time.time()

        if multi_objective_fitness:
            populations = []
            for island in range(len(cum_sum_butlast)):
                populations.append(Population(pop_size=pop_size, mating_pool_size=mating_pool_size, elite_size=elite_size, mutation_rate=mutation_rate, multi_objective_fitness=multi_objective_fitness, island_number=island))

            best_individual = Individual()
            best_fitness = 0
            best_generation = 0

            for i in range(0, self.generations):
                for pop in populations:
                    for ind in pop.population:
                        fitness = ind.calculate_fitness()
                        if fitness > best_fitness:
                            best_individual.individual = ind.individual.copy()
                            best_fitness = fitness
                            best_generation = i
                    pop.breedPopulation()

                if best_individual.isSolved():
                    print()
                    print("Nurikabe Solved in", time.time() - startTime, "seconds!")
                    print("Generation ", i, ": Best Fitness = ", best_fitness)
                    print("Best Individual: ", best_individual.individual)
                    print("Connected Islands: ", best_individual.findConnected())
                    print("Connected Oceans: ", best_individual.findConnectedOcean())
                    best_individual.printAsMatrix()
                    break

                if i % self.print_interval == 0:
                    print()
                    print("Generation ", i, ": Best Fitness = ", best_fitness)
                    print("Best Individual: ", best_individual.individual)
                    print("Connected Islands: ", best_individual.findConnected())
                    print("Connected Oceans: ", best_individual.findConnectedOcean())
                    best_individual.printAsMatrix()

                self.breedPopulations(populations)

        else:
            population = Population(
                pop_size=pop_size, mating_pool_size=mating_pool_size, elite_size=elite_size, mutation_rate=mutation_rate)

            best_individual = Individual()
            best_fitness = 0
            best_generation = 0
            avg_fitness = 0

            for i in range(0, self.generations):
                for ind in population.population:
                    # print("IND DIFF: ")
                    # ind.printAsMatrix()
                    # print()
                    # print("Individual: ", ind.individual)
                    fitness = ind.calculate_fitness()
                    avg_fitness += fitness
                    if fitness > best_fitness:
                        best_individual.individual = ind.individual.copy()
                        best_fitness = fitness
                        best_generation = i

                if best_individual.isSolved():
                    print()
                    print("Nurikabe Solved in", time.time() - startTime, "seconds!")
                    print("Generation ", i, ": Best Fitness = ", best_fitness)
                    print("Best Individual: ", best_individual.individual)
                    print("Connected Islands: ", best_individual.findConnected())
                    print("Connected Oceans: ", best_individual.findConnectedOcean())
                    best_individual.printAsMatrix()
                    break

                if i % self.print_interval == 0:
                    print()
                    print("Average Fitness: ", avg_fitness/pop_size)
                    print("Generation ", i, ": Best Fitness = ", best_fitness)
                    print("Best Individual: ", best_individual.individual)
                    print("Connected Islands: ", best_individual.findConnected())
                    print("Connected Oceans: ", best_individual.findConnectedOcean())
                    print("# Islands Isolated: ", best_individual.isIsolated())
                    print("Islands Not Isolated: ", best_individual.islandsNotIsolated())
                    best_individual.fixRange()
                    best_individual.printAsMatrix()

                avg_fitness = 0

                population.breedPopulation()

    def breedPopulations(self, populations):
        random.shuffle(populations)
        for x in range(len(populations)-1):
            cur_pop = populations[x]
            next_pop = populations[x+1]

            ind_num = 0
            for ind in populations[x].population:
                breed_chance = random.random()
                if breed_chance < 0.5:
                    ind = populations[x+1].population[ind_num]
                else:
                    populations[x].population[ind_num] = ind
                ind_num += 1



class Population():

    def __init__(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False, island_number=-1):

        self.population = []

        self.pop_size = pop_size
        self.mating_pool_size = mating_pool_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.multi_objective_fitness = multi_objective_fitness
        self.island_number = island_number

        for _ in range(pop_size):
            self.population.append(Individual(multi_objective_fitness))

    # - Selecting a random sample (like 100), then evaluating fitness and sorting them from best to worst.
    def getMatingPool(self):
        mating_pool = []

        sample = random.sample(self.population, k=self.mating_pool_size)

        if self.multi_objective_fitness:
            for individual in sample:
                mating_pool.append([individual, individual.calculate_fitness()])
        else:
            for individual in sample:
                mating_pool.append([individual, individual.calculate_fitness()])

        sorted_pool = sorted(mating_pool, key=lambda x: x[1], reverse=True)
        only_mates = [x for x, y in sorted_pool]

        # for x,y in sorted_pool:
            # print(x.individual, y)
        return only_mates

    # Using tournament selection, a random sample of elite size is taken
    # The elites are sorted by their fitness
    def breed(self, mating_pool):
        children = []
        elites = []

        # Maintain some elites based on elite size
        # Warning !! Mating pool must be larger than elite size.
        for i in range(self.elite_size):
            children.append(mating_pool[i])
            elites.append(mating_pool[i])

        # Pick a random parent from the list of elites
        # Pick another parent from the mating pool
        for i in range(self.mating_pool_size - self.elite_size):
            random_parent1 = random.randint(0, self.elite_size-1)
            random_parent2 = random.randint(0, self.mating_pool_size-1)
            
            random_chance = random.random()
            if random_chance < 0.5:
                parent1 = elites[random_parent1]
            else:
                parent1 = mating_pool[random.randint(0, self.mating_pool_size-1)]
            parent2 = mating_pool[random_parent2]

            # Find matching coordinates between two parents (Except the center coords)
            matching_coords = [
                x for x in parent1.individual if x in parent2.individual and x not in center_coords_keys]

            # Randomly select two island coordinates
            try:
                random_coords = random.sample(matching_coords, k=2)

                p2_a_index = parent2.index(random_coords[0])
                p2_b_index = parent2.index(random_coords[1])

                child = parent2.copy()
                child[p2_a_index] = random_coords[1]
                child[p2_b_index] = random_coords[0]

                children.append(child)
            except:
                pass

            children.append(Individual())

        for i in range(self.pop_size - self.mating_pool_size):
            children.append(Individual())

        return children

    # Moves a randomly selected island from Parent 1 and puts it into parent 2.
    # Anything that was in parent 1 that's found in parent 2 will be replaced with the remainders.
    # Added ocean swapping as well.
    def single_island_crossover(self, mating_pool):
        children = []
        elites = []

        # Maintain some elites based on elite size
        for i in range(self.elite_size):
            children.append(mating_pool[i])
            elites.append(mating_pool[i])

        for i in range(self.mating_pool_size - self.elite_size):
            # Find some random parents
            p1_random: Individual = random.choice(elites)
            p2_random: Individual = random.choice(mating_pool)

            # random_island_range picks a random range like [0,5] or [5,10]
            p1_island_range = p1_random.random_island_range()

            # Finds the actual elements in the range and converts it to a set
            if len(p1_island_range) != 1:
                p1_toSet = set(
                    p1_random.individual[p1_island_range[0]+1:p1_island_range[1]])
                p2_toSet = set(
                    p2_random.individual[p1_island_range[0]+1:p1_island_range[1]])
                avoid_range = range(p1_island_range[0] + 1, p1_island_range[1])
            else:
                p1_toSet = set(
                    p1_random.individual[p1_island_range[0]:list_size])
                p2_toSet = set(
                    p2_random.individual[p1_island_range[0]:list_size])
                avoid_range = range(p1_island_range[0], list_size)

            # DEBUGGING
            # print("P1: ", p1_random.individual)
            # print("P2: ", p2_random.individual)

            # Maintain p1_toSet but as a list, so we can use it for easy comparison
            p1_toList = list(p1_toSet)
            # Subtracting the p1 set from the p2 set and making sure we don't lose any coordinates
            remainders = list(p2_toSet - p1_toSet)

            # DEBUGGING
            # print("p1_toList: ", p1_toList)
            # print("Remainders: ", remainders)

            child = Individual()
            child.individual = p2_random.individual.copy()
            inner = 0

            for i in range(list_size):
                if i in avoid_range:
                    child.individual[i] = p1_toList[inner]
                    inner += 1
                else:
                    if child.individual[i] in p1_toList:
                        child.individual[i] = remainders.pop()

            # Debugging
            # print("Parent 1 : ", p1_random.individual)
            # print("Parent 2 : ", p2_random.individual)
            # print("Child    : ", child.individual)

            children.append(child)

        for i in range(self.pop_size - self.mating_pool_size):
            children.append(Individual())

        return children

    # Mutates all of the individual in a given population
    # In this case, it should be all the children
    # Randomly swaps a child with an ocean
    def mutate(self, children):
        for child in children:
            # Every child has a chance to have random coordinates swapped
            rand_chance = random.random()
            rand_chance_2 = random.random()
            squareOcean = child.findFirstOceanSquare()

            all_one_main_islands = [x for x in center_coords if center_coords[x] == 1]
            random_single_isolate = random.choice(all_one_main_islands) if all_one_main_islands else False

            if rand_chance < self.mutation_rate:
                swaps_at_a_time = random.randint(1, max_swaps)
                for _ in range(swaps_at_a_time):
                    random_land = random.randint(0, max_islands-1)
                    while random_land in cum_sum_butlast:
                        random_land = random.randint(0, max_islands-1)
                    random_ocean = random.randint(max_islands, list_size-1)
                    temp_coord = child.individual[random_land]
                    child.individual[random_land] = child.individual[random_ocean]
                    child.individual[random_ocean] = temp_coord
            else:
                child.propogationMutation()
                if rand_chance_2 < self.mutation_rate:
                    child.fixRange()
                else:
                    child.mutateRange()

            # Fixing a random coordinate in a square ocean
            if(squareOcean != 0):
                child.fixASquare(squareOcean)

            if random_single_isolate:
                child.isolateSingleIsland(random_single_isolate)

        return children

    def breedPopulation(self):
        mating_pool = self.getMatingPool()
        children = self.single_island_crossover(mating_pool)
        mutated = self.mutate(children)
        self.population = mutated
        return self

    # The tool to print an individual in a population as a matrix
    # 0 representing water, 1-X representing islands
    # Example:
    # Population #0:
    # [1, 0, 0, 1, 0]
    # [0, 2, 1, 0, 0]
    # [3, 1, 0, 3, 0]
    # [1, 0, 0, 0, 0]
    # [0, 0, 0, 0, 0]

    # Convert Individual to a matrix for visualization, Takes an index (int) of a population
    def printAsMatrix(self, index):
        # Initializing grid
        grid = [[0]*grid_size for i in range(grid_size)]

        # islandNumber is used to mark the grid
        islandNumber = 1
        # Using cumsum to get the indices of the individual to extract
        for i in range(len(cum_sum)-1):
            for x, y in self.population[index].individual[cum_sum[i]:cum_sum[i+1]]:
                # Assign the value in the grid
                grid[x][y] = islandNumber
            islandNumber += 1

        for _ in grid:
            print(_)


class Individual():

    # Structure of the individual is just this --> [(x,y), (x,y), ... (x,y)]
    # Randomly creates an individual with dedicated island indices - based off center_coords
    def __init__(self, multi_objective_fitness=False):

        self.individual = []

        self.multi_objective_fitness = multi_objective_fitness

        # ind keeps track of the index of the cum_sum_butlast
        isl = 0

        # Copy the list of valid coords and shuffle them in place
        random_valid_coords = valid_coords.copy()
        random.shuffle(random_valid_coords)

        # List building: Loop through size of the list ... (i = 0..24 for a 5x5)
        for i in range(list_size):

            # isl is initialized outside of the loop b/c it keeps track of the index of cum_sum_butlast
            # cum_sum_butlast is a list of indices where an island is supposed to start.
            if isl < len(cum_sum_butlast) and i == cum_sum_butlast[isl]:
                self.individual.append(center_coords_keys[isl])
                isl += 1
            else:
                # Pop & append a coordinate from the newly created randomized list of valid coords to the individual.
                self.individual.append(random_valid_coords.pop())

        self.ocean_start_index = cum_sum[-1]
        self.empty_list = [[0 for x in range(grid_size)] for y in range(grid_size)]

        # Create a list of possible squares. from 1 to -2, 2 to -1 for x and y 
        # This is makes searching for ocean squares less costly
        self.squares = []
        for i in range(grid_size-1):
            for j in range(grid_size-1):
                self.squares.append(((i,j),(i+1,j),(i,j+1),(i+1,j+1)))

    # FITNESS FUNCTIONS SUBJECT TO CHANGE!!!
    # Just a regular fitness function

    def calculate_fitness(self, island_focus=-1):
        total_fitness = 0.0

        # isIsolated() will return a value indicating how many good (or isolated) islands there are.
        # A perfectly fit individual will have a fitness equal to the length.

        if self.allInRange():
            pass
        else:
            return 0

        oceans_fitness = self.connectedFitnessOcean()
        isolation_fitness = self.isIsolated()

        if oceans_fitness == max_waters:
            total_fitness += max_waters
        else:
            return oceans_fitness
        
        if self.isOceanSquare():
            return oceans_fitness - self.numOceanSquares()*4

        total_fitness += isolation_fitness
        if isolation_fitness == len(center_coords):
            pass
        else:
            return total_fitness

        if island_focus != -1:
            total_fitness += self.connectedFitness()
        else:
            total_fitness += self.connectedFitness()

        return total_fitness

    # focus_island should be an island index that we want to focus on
    # We can also specify a weight to the fitness
    # The additional args are for multi objective fitness
    def calculate_overall_fitness(self):
        total_fitness = 0

        # TODO

        return total_fitness

    # isAdj checks two coordinates to see if theyre adjacent and returns a boolean
    def isAdj(self, coord1, coord2):
    # For clarity
        x1,y1 = coord1
        x2,y2 = coord2
        # return (abs(x1-x2) <= 1 and abs(y1-y2) <= 1)
        return (abs(x2-x1) + abs(y2-y1) == 1)

    # checks a list to see if any is adj
    def isAdjinList(self, coordlist, coord):
        for coordinate in coordlist:
            if self.isAdj(coordinate,coord):
                return True
        return False

    # returns the coordinate from coordlist1 that is adj to coordlist2 otherwise returns 0
    def coordAdjbetweenTwoLists(self,coordlist1, coordlist2):
        for coord1 in coordlist1:
            for coord2 in coordlist2:
                if(self.isAdj(coord1,coord2)):
                    return coord1
        return 0

    # This puts everything in seperate lists. i could have put this in find connected, but
    # this is easier to understand
    # this will return as many lists as there are islands
    def prepareIslandLists(self):
        tempList = []
        combinedLists = []
        for i in range(len(cum_sum)-1):
            for coord in self.individual[cum_sum[i]:cum_sum[i+1]]:
                tempList.append(coord)
            combinedLists.append(tempList)
            tempList = []
        return combinedLists

    # Returns a list of each connected island
    # Eg: 4 islands will return something like [[(0,3),(1,3)],[(2,1)],[(2,3),(3,3)],[(4,1)]]
    def findConnected(self):
        # Preparing the list of islands for calculation
        islands = self.prepareIslandLists()
        # Initializing a list of connected Island coordinates
        connectedIslands = []
        # Initializing the list that will be used to check for adjacencies
        coordsAdjinclCenter = []
        # This Boolean will specify when to stop searching for adjacencies (when no match is found)
        searching = True

        for island in islands:
            # Add the center and remove it from the island
            coordsAdjinclCenter.append(island.pop(0))
            while(searching):
                # Compare coordsAdjinclCenter with island. If any adj is found, add it
                # to coordsAdjinclCenter and remove it from the island

                # Temporary variable to reduce cost
                adjCoord = self.coordAdjbetweenTwoLists(island,coordsAdjinclCenter)
                # If no adj is found, coordAdjbetweenTwoLists will return 0
                if(adjCoord != 0):
                    coordsAdjinclCenter.append(island.pop(island.index(adjCoord)))
                else:
                    # No matches found, stop the search
                    searching = False

            connectedIslands.append(coordsAdjinclCenter)
            coordsAdjinclCenter = []
            searching = True
        return connectedIslands

    # Same as findConnected() but specifically for oceans
    def findConnectedOcean(self):
        ocean = self.individual[self.ocean_start_index:len(self.individual)]
        # Initializing a list of connected Island coordinates
        connectedOceans = []
        # Initializing the list that will be used to check for adjacencies
        coordsAdjinclCenter = []
        # This Boolean will specify when to stop searching for adjacencies (when no match is found)
        searching = True

        coordsAdjinclCenter.append(ocean.pop(0))
        while(searching):
            adjCoord = self.coordAdjbetweenTwoLists(ocean,coordsAdjinclCenter)
            # print("adJCOORD ", adjCoord)
            if(adjCoord != 0):
                coordsAdjinclCenter.append(ocean.pop(ocean.index(adjCoord)))
            else:
                # No matches found, stop the search
                searching = False

        connectedOceans.append(coordsAdjinclCenter)
        return connectedOceans

    # TODO: check for the longest length ocean, not just the ones adj to the first
    def findConnectedOceans2(self):
        searching = True
        coordsAdjtoFirst = []
        ocean = self.individual[cum_sum[-1]:]
        coordsAdjtoFirst.append(ocean.pop(0))
        while(searching):
            adjCoord = self.coordAdjbetweenTwoLists(ocean,coordsAdjtoFirst)
            if(adjCoord != 0):
                coordsAdjtoFirst.append(ocean.pop(ocean.index(adjCoord)))
            else:
                searching = False
        return coordsAdjtoFirst
    
    def connectedOceanFitness2(self):
        bestOceanSize = list_size - cum_sum[-1]
        connectedOcean = self.findConnectedOceans2()
        if(len(connectedOcean) == bestOceanSize):
            # double the points if its the right size
            return bestOceanSize * 2
        return len(connectedOcean)

    # Returns whether or not there is a square in the ocean
    def isOceanSquare(self):
        for square in self.squares:
            if(set(square).issubset(self.individual[cum_sum[-1]:])):
                return True
        return False
    
    def findFirstOceanSquare(self):
        for square in self.squares:
            if(set(square).issubset(self.individual[cum_sum[-1]:])):
                return square
        return 0

    def numOceanSquares(self):
        ct = 0
        for square in self.squares:
            if(set(square).issubset(self.individual[cum_sum[-1]:])):
                ct += 1
        return ct
    
    # returns a list of adj coordinates
    # size is the length of the side of the grid eg. (5x5) = 5
    def adjCoords(self, coord):
        adjCoords = []
        x,y = coord
        if (x > 0):
            adjCoords.append((x-1,y))
        if (y > 0):
            adjCoords.append((x,y-1))
        if (x < grid_size-1):
            adjCoords.append((x+1,y))
        if(y < grid_size-1):
            adjCoords.append((x,y+1))
        return adjCoords

    # Given a coordinate returns whether or not it is an island
    def isIsland(self,coord):
        if(self.individual.index(coord) < cum_sum[-1]):
            return True
        return False
    
    def isolateSingleIsland(self, coord):
        # Check all the adj if theyre islands
        adjIslands = []
        for coordinate in self.adjCoords(coord):
            if(self.isIsland(coordinate)):
                adjIslands.append(coordinate)
        
        # For each island found, swap it with a random ocean
        for island in adjIslands:
            randomOceanIndex = self.individual.index(random.choice(self.individual[cum_sum[-1]:]))
            tempIsland = island
            self.individual[self.individual.index(island)] = self.individual[randomOceanIndex]
            self.individual[randomOceanIndex] = tempIsland

    # Checks to see if coordinate 1 is in range of coordinate 2, based on center
    # value length 
    def inRange(self, centerValue, coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        distance = abs(x2-x1) + abs(y2-y1)
        if (distance <= centerValue):
            return True
        return False


    def connectedFitness(self):
        connectedIslands = self.findConnected()
        # give a big fitness bonus if the size of the island is the correct size
        # first we have to identify the size each island must be
        # Using list comprehension, I can use zip to combine the same list to subtract
        # each value next to each other to give me the island sizes
        bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
        bestScore = max(bestIslandSizes)

        # the sum of the list created from: if the island is > than the size, -1 point.
        # if it is an incorrect size, then just add the the size of the island
        # if it is a correct size then add the size of the island with a bonus 3. (larger correct islands give bigger points)

        # Original Code Below
        # connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes
        # else len(cIsland)+3 for (cIsland, sizes) in zip(connectedIslands, bestIslandSizes)])

        # Changed for some testing
        connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes
        else bestScore for (cIsland, sizes) in zip(connectedIslands, bestIslandSizes)])

        return connectedFitness

    def connectedFitnessOcean(self):
        connectedOceans = self.findConnectedOcean()[0]

        # A different connectedFitness for testing
        connectedFitness = max_waters if len(connectedOceans) == max_waters else len(connectedOceans)
        
        # Update: i realized that a bonus 3 would mean that small islands with correct size would not be valuable
        # so instead, full size islands will get a max fitness, which is the highest island cost
        # connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes 
        # else bestScore for (cIsland, sizes) in zip(connectedOceans, bestIslandSizes)])
        return connectedFitness

    # Specify which island gets a weight
    def connectedFitnessWeighted(self, island_number):
        connectedIsland = self.findConnected()[island_number]
        bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
        bestIslandSize = bestIslandSizes[island_number]
        islandFitness = bestIslandSize if len(connectedIsland) == bestIslandSize else len(connectedIsland) if len(connectedIsland) < bestIslandSize else len(connectedIsland) + (bestIslandSize - len(connectedIsland))
        return islandFitness - 1

    # This is the main isIsolated function that is currently in use.
    def isIsolated(self):
        # The island's adjacencies should only contain itself or an ocean.
        # Keeps track of iterations
        isl = 0

        # Incorporated a fitness value
        fitness_val = 0

        # For each coordinate in an island, the coordinate's adjacent position must be within itself or the ocean.
        for island_start in cum_sum_butlast:
            island_end = cum_sum[isl+1]

            # island is a list or splice of coordinates corresponding to an island
            island = self.individual[island_start:island_end]
            other_islands = list(set(self.individual[0:cum_sum[-1]])-set(island))

            # An island will stay "Good" if it's isolated.
            good_island = True

            ## TODO ##
            # Extremely inefficient!! MAKE IT BETTER!
            all_adjacents = []
            for coord in island:
                adjacents = adjacencies[coord]
                for a in adjacents:
                    all_adjacents.append(a)
            all_adjacents_no_dupes = set(all_adjacents)
            for coord in island:
                if coord not in all_adjacents_no_dupes and len(island) != 1:
                    good_island = False
            for a in all_adjacents_no_dupes:
                if a in other_islands:
                    good_island = False

            if good_island:
                fitness_val += 1

            isl += 1

        return fitness_val

    # Returns a list with each index corresponding to an island.
    # [-1, 1, -1] means the first island is not isolated, while the second one is. Third island is not isolated.
    def islandsNotIsolated(self):
        # The island's adjacencies should only contain itself or an ocean.
        # Keeps track of iterations
        isl = 0

        # Incorporated a fitness value
        isolated_islands = []

        # For each coordinate in an island, the coordinate's adjacent position must be within itself or the ocean.
        for island_start in cum_sum_butlast:
            island_end = cum_sum[isl+1]

            # island is a list or splice of coordinates corresponding to an island
            island = self.individual[island_start:island_end]
            other_islands = list(set(self.individual[0:cum_sum[-1]])-set(island))

            # An island will stay "Good" if it's isolated.
            good_island = True

            ## TODO ##
            # Extremely inefficient!! MAKE IT BETTER!
            all_adjacents = []
            for coord in island:
                adjacents = adjacencies[coord]
                for a in adjacents:
                    all_adjacents.append(a)
            all_adjacents_no_dupes = set(all_adjacents)
            for coord in island:
                if coord not in all_adjacents_no_dupes and len(island) != 1:
                    good_island = False
            for a in all_adjacents_no_dupes:
                if a in other_islands:
                    good_island = False

            if good_island:
                pass
            else:
                isolated_islands.append(island)

            isl += 1

        return isolated_islands

    # Returns a random island range
    # Also does oceans now
    def random_island_range(self):
        island_start_index = random.choice(range(len(cum_sum_withmax)))
        # print("Island Indices being swapped: ",
        #       cum_sum[island_start_index:island_start_index+2])
        return cum_sum_withmax[island_start_index:island_start_index + 2]

    # Finds an island that has incomplete connections
    # Returns the island number (or index)
    def shortIsland(self):
        islands = self.findConnected()

        island_number = 0
        for island in islands:
            if len(island) != center_coords_vals[island_number]:
                return island_number,island
            else:
                island_number += 1
        return -1,[]

    # Finds out where an island can be swapped with ocean to achieve full connection
    def addToShortIsland(self):
        # The starting index for cum_sum of the island, and the actual connected islands
        short_island_index,connected_islands = self.shortIsland()

        if short_island_index == -1:
            return ()
        # The full island
        short_island = self.individual[cum_sum[short_island_index]:cum_sum[short_island_index+1]]

        # Find coordinates that are not connected
        replacement_coordinates = set(short_island) - set(connected_islands)

        # Find any possible adjacent oceans that these coordinates can take around the actual connected
        coords_all_adjacent = set([x for sub in [adjacencies[coord] for coord in connected_islands] for x in sub])
        valid_adjacents_in_ocean = coords_all_adjacent.intersection(set(self.individual[self.ocean_start_index:len(self.individual)]))

        return (list(replacement_coordinates), list(valid_adjacents_in_ocean))

    def propogationMutation(self):
        if len(self.addToShortIsland()) != 0:
            replacement_coordinates, valid_adjacents_in_ocean = self.addToShortIsland()
            random.shuffle(valid_adjacents_in_ocean)

            for coord in replacement_coordinates:
                my_index = self.individual.index(coord)
                if valid_adjacents_in_ocean:
                    ocean_index = self.individual.index(valid_adjacents_in_ocean.pop())
                    temp_coord = coord
                    self.individual[my_index] = self.individual[ocean_index]
                    self.individual[ocean_index] = temp_coord
        else:
            pass

    def printAsMatrix(self):

        island_number = 1
        ct = 0
        for x,y in self.individual:
            if ct in cum_sum_butlast[1:]:
                island_number += 1
            elif ct >= cum_sum[-1]:
                island_number = 0

            self.empty_list[x][y] = island_number

            ct += 1

        for row in self.empty_list:
            print(row)

    def isSolved(self):
        bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
        bestScore = max(bestIslandSizes)
        largest_fitness_possible = len(center_coords) + max_waters + (bestScore * len(center_coords))
        if self.isIsolated() == len(center_coords) and self.connectedFitnessOcean() == max_waters and not self.isOceanSquare() and self.calculate_fitness() == largest_fitness_possible:
            return True

        # squares = [(x,y), (x,y), (x,y), (x,y)]
    def fixASquare(self, square):
        # Ocean
        if (square != 0):
            # print("test")
            random_ocean_coord = random.choice(square)
            # print("random ocean coord: ", random_ocean_coord)
            random_ocean_index = self.individual.index(random_ocean_coord)
            # print("random ocean index: ", random_ocean_index)
            closest_coord_range = self.closestMainIsland(random_ocean_coord)

            if closest_coord_range:
                # print("closest coord range: ", closest_coord_range)
                # One of the closest lands
                random_land_index = random.choice(closest_coord_range)
                # print("random land index: ", random_land_index)
                random_land_coord = self.individual[random_land_index]
                self.individual[random_land_index] = random_ocean_coord
                self.individual[random_ocean_index] = random_land_coord

    def closestMainIsland(self, coord):
        coordX, coordY = coord
        closest_distance = sys.maxsize
        closest_coord_range = []
        next_index = 1
        for main_coord_index in cum_sum_butlast:
            mainX, mainY = self.individual[main_coord_index]
            # distance = math.sqrt(abs(coordX - mainX)**2 + abs(coordY - mainY)**2) #Euclidian
            distance = abs(coordX - mainX) + abs(coordY - mainY) #manhattan
            if distance < closest_distance:
                closest_distance = distance
                closest_coord_range = range(main_coord_index+1, cum_sum[next_index])
            next_index += 1
        return closest_coord_range
    
    def fixRange(self):
        for i in range(len(cum_sum_butlast)):
            center = self.individual[cum_sum[i]]
            max_size = center_coords[center]
            for coord in self.individual[cum_sum[i]+1:cum_sum[i+1]]:
                if max_size != 1:
                    if self.inRange(max_size, center, coord):
                        pass
                    else:
                        in_range_oceans = [x for x in self.individual[cum_sum[-1]:] if self.inRange(max_size, center, x)]
                        if in_range_oceans:
                            random_ocean = random.choice(in_range_oceans)
                            rand_ocean_index = self.individual.index(random_ocean)
                            coord_index = self.individual.index(coord)
                            
                            self.individual[rand_ocean_index] = coord
                            self.individual[coord_index] = random_ocean
    
    def allInRange(self):
        for i in range(len(cum_sum_butlast)):
            center = self.individual[cum_sum[i]]
            max_size = center_coords[center]
            for coord in self.individual[cum_sum[i]+1:cum_sum[i+1]]:
                if max_size != 1:
                    if self.inRange(max_size, center, coord):
                        pass
                    else:
                        return False
        return True

    def mutateRange(self):
        for i in range(len(cum_sum_butlast)):
            center = self.individual[cum_sum[i]]
            max_size = center_coords[center]
            coords = self.individual[cum_sum[i]+1:cum_sum[i+1]]
            if coords:
                coord = random.choice(coords)
                if max_size != 1:
                    in_range_oceans = [x for x in self.individual[cum_sum[-1]:] if self.inRange(max_size, center, x)]
                    if in_range_oceans:
                        random_ocean = random.choice(in_range_oceans)
                        rand_ocean_index = self.individual.index(random_ocean)
                        coord_index = self.individual.index(coord)
                                    
                        self.individual[rand_ocean_index] = coord
                        self.individual[coord_index] = random_ocean




def main():
    nurikabe = NurikabeGA(grid_size=grid_size, center_coords=center_coords, generations=100000, print_interval=5)
    nurikabe.geneticAlgorithm(
        pop_size=2000, mating_pool_size=1000, elite_size=100, mutation_rate=0.5, multi_objective_fitness=False)


    return 0


if __name__ == "__main__":
    main()
