import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class NSGAII:

    class _Individual:

        def __init__(self):

            self.position = None
            self.cost = None
            self.domination_set = None
            self.dominated_count = None
            self.rank = None
            self.crowding_distance = None

    def __init__(self,
                 number_variables,
                 min_range_variables,
                 max_range_variables,
                 cost_function,
                 max_iteration,
                 number_population,
                 crossover_percentage,
                 mutation_percentage,
                 mutation_rate):

        self._number_variables = number_variables
        self._min_range_variables = min_range_variables
        self._max_range_variables = max_range_variables
        self._cost_fn = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_population
        self._population_main = None
        self._crossover_percentage = crossover_percentage
        self._number_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_population / 2))
        self._population_crossover = None
        self._mutation_percentage = mutation_percentage
        self._number_mutation = int(np.ceil(self._mutation_percentage * self._number_population))
        self._population_mutation = None
        self._mutation_rate = mutation_rate
        self._pareto_front = None

    def _initialize_population(self):

        population = [self._Individual() for _ in range(self._number_population)]

        for i in range(self._number_population):

            population[i].position = np.random.uniform(self._min_range_variables,
                                                       self._max_range_variables,
                                                       (1, self._number_variables))

            population[i].cost = self._cost_fn(population[i].position)

        return population

    @staticmethod
    def _dominates(population1_cost, population2_cost):

        return np.all(population1_cost <= population2_cost) and np.any(population1_cost < population2_cost)

    def _non_dominated_sorting(self, population):

        pareto_front = {0: []}

        for i in range(len(population)):

            population[i].dominated_count = 0
            population[i].domination_set = []

        for i in range(len(population)):

            for j in range(i + 1, len(population)):

                if self._dominates(population[i].cost, population[j].cost):

                    population[i].domination_set.append(j)
                    population[j].dominated_count += 1

                elif self._dominates(population[j].cost, population[i].cost):

                    population[j].domination_set.append(i)
                    population[i].dominated_count += 1

            if population[i].dominated_count == 0:

                population[i].rank = 0
                pareto_front[0].append(i)

        counter = 0
        while True:

            Q = []
            for i in pareto_front[counter]:
                for j in population[i].domination_set:

                    population[j].dominated_count -= 1
                    if population[j].dominated_count == 0:

                        population[j].rank = counter + 1
                        Q.append(j)

            if len(Q) == 0:
                break
            counter += 1
            pareto_front[counter] = Q

        return population, pareto_front

    @staticmethod
    def _crowding_distance(population, pareto_front):

        for front in pareto_front:

            if len(pareto_front[front]) <= 2:

                for i in pareto_front[front]:

                    population[i].crowding_distance = np.random.uniform(10, 20)

            else:

                population_costs = np.hstack([population[i].cost for i in pareto_front[front]])

                population_costs_argsort = np.argsort(population_costs, axis=1)

                first_index = pareto_front[front][int(population_costs_argsort[0, 0])]
                last_index = pareto_front[front][int(population_costs_argsort[0, -1])]

                population[first_index].crowding_distance = np.random.uniform(10, 20)
                population[last_index].crowding_distance = np.random.uniform(10, 20)

                distance_max = np.abs(population[first_index].cost - population[last_index].cost)

                for i in range(len(pareto_front[front])):

                    if i != 0 and i != len(pareto_front[front]) - 1:

                        prev_index = pareto_front[front][int(population_costs_argsort[0, i - 1])]
                        current_index = pareto_front[front][int(population_costs_argsort[0, i])]
                        next_index = pareto_front[front][int(population_costs_argsort[0, i + 1])]

                        distance = np.abs(population[prev_index].cost - population[next_index].cost)

                        population[current_index].crowding_distance = np.sum(np.divide(distance, distance_max))

        return population

    @staticmethod
    def _sort(population, pareto_front):

        population_sorted = []
        for front in pareto_front:

            pop_argsort = np.argsort([population[j].crowding_distance for j in pareto_front[front]])[::-1]

            pop_current = [population[pareto_front[front][int(i)]] for i in pop_argsort]

            population_sorted.extend(pop_current)

        return population_sorted

    def _binary_tournament_selection(self):

        indices = [int(i) for i in np.random.choice(range(self._number_population), 2, replace=False)]
        index1, index2 = int(min(indices)), int(max(indices))

        if self._population_main[index1].rank < self._population_main[index2].rank:
            return index1
        elif self._population_main[index2].rank < self._population_main[index1].rank:
            return index2
        else:
            if self._population_main[index1].crowding_distance > self._population_main[index2].crowding_distance:
                return index1
            elif self._population_main[index2].crowding_distance > self._population_main[index1].crowding_distance:
                return index2

    def _apply_crossover(self, population1, population2):

        position1 = population1.position
        position2 = population2.position

        alpha = np.random.uniform(-0.1, 1.1, position1.shape)

        offspring1 = np.multiply(alpha, position1) + np.multiply(1 - alpha, position2)
        offspring2 = np.multiply(alpha, position2) + np.multiply(1 - alpha, position1)

        offspring1 = np.clip(offspring1, self._min_range_variables, self._max_range_variables)
        offspring2 = np.clip(offspring2, self._min_range_variables, self._max_range_variables)

        return offspring1, offspring2

    def _apply_mutation(self, population):

        position = population.position

        number_mutants = int(np.ceil(self._mutation_rate * self._number_variables))
        mutate_cells = [int(i) for i in np.random.choice(range(self._number_variables), number_mutants, replace=False)]

        offspring = np.copy(position)

        offspring[0, mutate_cells] = position[0, mutate_cells] + \
                                     (0.1 * (self._max_range_variables - self._min_range_variables)) * \
                                     np.random.randn(number_mutants)

        offspring = np.clip(offspring, self._min_range_variables, self._max_range_variables)

        return offspring


    def run(self):

        tic = time.time()

        self._population_main = self._initialize_population()

        self._population_main, self._pareto_front = self._non_dominated_sorting(self._population_main)

        self._population_main = self._crowding_distance(self._population_main, self._pareto_front)

        self._population_main = self._sort(self._population_main, self._pareto_front)

        for iter_main in range(self._max_iteration):

            self._population_crossover = [self._Individual() for _ in range(self._number_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._number_mutation)]

            for iter_crossover in range(0, self._number_crossover, 2):

                parent1_index = self._binary_tournament_selection()
                parent2_index = self._binary_tournament_selection()
                while parent1_index == parent2_index:
                    parent2_index = self._binary_tournament_selection()

                self._population_crossover[iter_crossover].position, \
                self._population_crossover[iter_crossover + 1].position = \
                self._apply_crossover(self._population_main[parent1_index], self._population_main[parent2_index])

                self._population_crossover[iter_crossover].cost = \
                self._cost_fn(self._population_crossover[iter_crossover].position)

                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_fn(self._population_crossover[iter_crossover + 1].position)


            for iter_mutation in range(self._number_mutation):

                parent_index = self._binary_tournament_selection()

                self._population_mutation[iter_mutation].position = \
                self._apply_mutation(self._population_main[parent_index])

                self._population_mutation[iter_mutation].cost = \
                self._cost_fn(self._population_mutation[iter_mutation].position)

            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)

            self._population_main, self._pareto_front = self._non_dominated_sorting(self._population_main)

            self._population_main = self._crowding_distance(self._population_main, self._pareto_front)

            self._population_main = self._sort(self._population_main, self._pareto_front)

            self._population_main = self._population_main[:self._number_population]

            self._population_main, self._pareto_front = self._non_dominated_sorting(self._population_main)

            self._population_main = self._crowding_distance(self._population_main, self._pareto_front)

            self._population_main = self._sort(self._population_main, self._pareto_front)

        toc = time.time()

        costs = np.hstack([self._population_main[i].cost for i in self._pareto_front[0]])

        os.makedirs("./figures", exist_ok=True)
        plt.figure(dpi=300, figsize=(10, 6))
        plt.scatter(costs[0, :], costs[1, :], marker="x", c="r", s=8)
        plt.xlabel("First Objective Function")
        plt.ylabel("Second Objective Function")
        plt.title("MOP2 Using Non-Dominated Sorting Genetic Algorithm - II", fontweight="bold")
        plt.savefig("./figures/pareto_front.png")

        return self._population_main, self._pareto_front, toc - tic
