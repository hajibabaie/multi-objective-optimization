from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class SPEAII:

    class _Individual:

        def __init__(self):

            self.position = None
            self.cost = None
            self.strength = None
            self.raw_fitness = None
            self.distances = None
            self.density = None
            self.fitness = None

    def __init__(self,
                 number_variables,
                 min_range_variables,
                 max_range_variables,
                 cost_function,
                 max_iteration,
                 number_population,
                 number_archive,
                 crossover_percentage,
                 mutation_rate):

        self._number_variables = number_variables
        self._min_range_variables = min_range_variables
        self._max_range_variables = max_range_variables
        self._cost_fn = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_population
        self._number_archive = number_archive
        self._crossover_percentage = crossover_percentage
        self._number_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_population / 2))
        self._number_mutation = self._number_population - self._number_crossover
        self._mutation_rate = mutation_rate
        self._k = int(np.sqrt(self._number_population + self._number_archive))
        self._population_main = [self._Individual() for _ in range(self._number_population)]
        self._population_crossover = None
        self._population_mutation = None
        self._population_total = None
        self._archive = []


    def _initialize_population(self):

        for i in range(self._number_population):

            self._population_main[i].position = np.random.uniform(self._min_range_variables,
                                                                  self._max_range_variables,
                                                                  (1, self._number_variables))

            self._population_main[i].cost = self._cost_fn(self._population_main[i].position)

    def _merge_population(self):
        self._population_total = []
        self._population_total.extend(copy.deepcopy(self._population_main))
        self._population_total.extend(copy.deepcopy(self._archive))

    @staticmethod
    def _dominate(population1_cost, population2_cost):

        return np.all(population1_cost <= population2_cost) and np.any(population1_cost < population2_cost)

    def _calc_strength_and_raw_fitness(self):

        for i in range(len(self._population_total)):

            self._population_total[i].strength = 0

        domination_matrix = np.zeros((len(self._population_total), len(self._population_total)))
        for i in range(len(self._population_total)):

            for j in range(len(self._population_total)):

                if self._dominate(self._population_total[i].cost, self._population_total[j].cost):

                    domination_matrix[i, j] = 1
                    self._population_total[i].strength += 1

                elif self._dominate(self._population_total[j].cost, self._population_total[i].cost):

                    domination_matrix[j, i] = 1
                    self._population_total[j].strength += 1

            self._population_total[i].raw_fitness = np.sum([self._population_total[int(k)].strength
                                                            for k in np.argwhere(domination_matrix[:, i]) != 0])

    def _calc_distances_density_info_and_fitness(self):

        costs = np.vstack([self._population_total[i].cost for i in range(len(self._population_total))])
        distances = np.sort(cdist(costs, costs, "seuclidean"), axis=0)
        for i in range(len(self._population_total)):
            self._population_total[i].distances = distances[:, i: i + 1]
            self._population_total[i].density = 1 / (self._population_total[i].distances[self._k, 0] + 2)
            self._population_total[i].fitness = self._population_total[i].raw_fitness + self._population_total[i].density

    def _sort(self):

        argsort = np.argsort([self._population_total[i].fitness for i in range(len(self._population_total))])
        self._population_total = [self._population_total[int(i)] for i in argsort]

    def _update_archive(self):

        number_non_dominated = len([pop for pop in self._population_total if pop.raw_fitness == 0.0])

        if number_non_dominated <= self._number_archive:

            self._archive = copy.deepcopy(self._population_total[:self._number_archive])

        else:

            self._archive = copy.deepcopy([pop for pop in self._population_total if pop.raw_fitness == 0.0])
            k = 1
            while len(self._archive) > self._number_archive:

                distances = np.hstack([self._archive[i].distances for i in range(len(self._archive))])
                while np.min(distances[k, :]) == np.max(distances[k, :]) and k < distances.shape[0]:
                    k += 1

                index_to_delete = int(np.argmin(distances[k, :]))
                self._archive.pop(index_to_delete)

    def _binary_tournament_selection(self):

        indices = [int(i) for i in np.random.choice(range(len(self._archive)), 2, replace=False)]
        index_f, index_s = min(indices), max(indices)
        if self._archive[index_f].fitness < self._archive[index_s].fitness:
            return index_f
        else:
            return index_s

    def _apply_crossover(self, population1, population2):

        position1 = population1.position
        position2 = population2.position

        alpha = np.random.uniform(-0.1, 1.1, position1.shape)

        offspring1 = np.multiply(alpha, position1) + np.multiply(1 - alpha, position2)
        offspring2 = np.multiply(1 - alpha, position2) + np.multiply(alpha, position1)

        offspring1 = np.clip(offspring1, self._min_range_variables, self._max_range_variables)
        offspring2 = np.clip(offspring2, self._min_range_variables, self._max_range_variables)

        return offspring1, offspring2

    def _apply_mutation(self, population):

        position = population.position

        number_mutants = int(np.ceil(self._mutation_rate * self._number_variables))

        mutated_cells = [int(i) for i in np.random.choice(range(self._number_variables), number_mutants, replace=False)]

        offspring = np.copy(position)

        offspring[:, mutated_cells] = position[:, mutated_cells] + (0.1 * (self._max_range_variables -
                                                                           self._min_range_variables) *
                                                                    np.random.randn(number_mutants))

        offspring = np.clip(offspring, self._min_range_variables, self._max_range_variables)

        return offspring

    def run(self):

        tic = time.time()

        self._initialize_population()

        for iter_main in range(self._max_iteration):

            self._merge_population()

            self._calc_strength_and_raw_fitness()

            self._calc_distances_density_info_and_fitness()

            self._sort()

            self._update_archive()


            self._population_crossover = [self._Individual() for _ in range(self._number_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._number_mutation)]
            for iter_c in range(0, self._number_crossover, 2):

                parent1 = self._binary_tournament_selection()
                parent2 = self._binary_tournament_selection()
                while parent1 == parent2:
                    parent2 = self._binary_tournament_selection()

                self._population_crossover[iter_c].position, \
                self._population_crossover[iter_c + 1].position = \
                self._apply_crossover(self._archive[parent1], self._archive[parent2])

                self._population_crossover[iter_c].cost = self._cost_fn(self._population_crossover[iter_c].position)
                self._population_crossover[iter_c+1].cost = self._cost_fn(self._population_crossover[iter_c+1].position)

            for iter_m in range(self._number_mutation):

                parent = self._binary_tournament_selection()

                self._population_mutation[iter_m].position = self._apply_mutation(self._archive[parent])
                self._population_mutation[iter_m].cost = self._cost_fn(self._population_mutation[iter_m].position)

            self._population_main = []
            self._population_main.extend(copy.deepcopy(self._population_crossover))
            self._population_main.extend(copy.deepcopy(self._population_mutation))

        toc = time.time()

        os.makedirs("./figures", exist_ok=True)

        costs = np.vstack([pop.cost for pop in self._archive])
        plt.figure(dpi=300, figsize=(10, 6))
        plt.scatter(costs[:, 0], costs[:, 1], marker="x", c="red", s=8)
        plt.xlabel("First Objective Function")
        plt.ylabel("Second Objective Function")
        plt.title("MOP2 Using Strength Pareto Evolutionary Algorithm - II", fontweight="bold")
        plt.savefig("./figures/pareto_front.png")

        return self._population_main, self._population_total, self._archive, toc - tic
