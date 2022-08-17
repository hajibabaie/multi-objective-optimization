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
            self.rank = None
            self.crowding_distance = None
            self.domination_set = None
            self.dominated_count = None

    def __init__(self,
                 number_variables,
                 min_range_variables,
                 max_range_variables,
                 cost_function,
                 max_iteration,
                 number_population,
                 crossover_percentage,
                 mutation_percentage,
                 mutation_rate,
                 ):

        self._number_variables = number_variables
        self._min_range_variables = min_range_variables
        self._max_range_variables = max_range_variables
        self._cost_function = cost_function
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

            population[i].cost = self._cost_function(population[i].position)

        return population

    @staticmethod
    def _non_dominated_sorting(population_main):

        def dominate(population_first, population_second):

            cost_first = population_first.cost
            cost_second = population_second.cost

            return np.all(cost_first <= cost_second) and np.any(cost_first < cost_second)

        pareto_front = {0: []}

        for i in range(len(population_main)):

            population_main[i].domination_set = []
            population_main[i].dominated_count = 0

        for i in range(len(population_main)):

            for j in range(i + 1, len(population_main)):

                if dominate(population_main[i], population_main[j]):

                    population_main[i].domination_set.append(j)
                    population_main[j].dominated_count += 1

                elif dominate(population_main[j], population_main[i]):

                    population_main[j].domination_set.append(i)
                    population_main[i].dominated_count += 1

            if population_main[i].dominated_count == 0:

                population_main[i].rank = 0
                pareto_front[0].append(i)

        counter = 0
        while True:

            Q = []

            for population_index in pareto_front[counter]:

                for population_dominated_index in population_main[population_index].domination_set:

                    population_main[population_dominated_index].dominated_count -= 1

                    if population_main[population_dominated_index].dominated_count == 0:

                        Q.append(population_dominated_index)
                        population_main[population_dominated_index].rank = counter + 1

            if len(Q) == 0:
                break

            counter += 1
            pareto_front[counter] = copy.deepcopy(Q)

        return population_main, pareto_front

    @staticmethod
    def _crowding_distance(population, pareto_front):

        for front in pareto_front:

            if len(pareto_front[front]) <= 2:

                for pop_index in pareto_front[front]:

                    population[pop_index].crowding_distance = np.random.uniform(10, 20)

            else:

                population_cost = np.hstack([population[i].cost for i in pareto_front[front]])
                population_cost_distance = np.zeros((int(population_cost.shape[0]), 1))

                for i in range(int(population_cost.shape[0])):

                    population_cost_distance[i, 0] = np.max(population_cost[i, :]) - np.min(population_cost[i, :])

                population_cost_argsort = [int(i) for i in np.argsort(population_cost[0, :])]

                for i in range(len(pareto_front[front])):

                    index = pareto_front[front][population_cost_argsort[i]]
                    if i == 0 or i == len(pareto_front[front]) - 1:

                        population[index].crowding_distance = np.random.uniform(10, 20)

                    else:

                        prev_index = pareto_front[front][population_cost_argsort[i - 1]]
                        next_index = pareto_front[front][population_cost_argsort[i + 1]]

                        distance = np.zeros((int(population_cost.shape[0]), 1))

                        for obj in range(int(population_cost.shape[0])):

                            distance[obj, 0] = population[prev_index].cost[obj, 0] - population[next_index].cost[obj, 0]
                            distance[obj, 0] = np.abs(distance[obj, 0])

                        population[index].crowding_distance = np.squeeze(np.sum(np.divide(distance, population_cost_distance)))

        return population


    @staticmethod
    def _sort(population, pareto_front):

        population_sorted = []

        for front in pareto_front:

            index_current = [int(i) for i in np.argsort([population[j].crowding_distance for j in pareto_front[front]])[::-1]]
            pop_sorted = [population[pareto_front[front][i]] for i in index_current]
            population_sorted.extend(pop_sorted)

        return population_sorted

    def _binary_tournament_selection(self):

        indices = [int(i) for i in np.random.choice(range(self._number_population), 2, replace=False)]
        first_index = min(indices)
        second_index = max(indices)

        if self._population_main[first_index].rank < self._population_main[second_index].rank:

            return first_index

        elif self._population_main[second_index].rank < self._population_main[first_index].rank:

            return second_index

        else:

            if self._population_main[first_index].crowding_distance > self._population_main[second_index].crowding_distance:

                return first_index

            elif self._population_main[second_index].crowding_distance > self._population_main[first_index].crowding_distance:

                return second_index


    def _apply_crossover(self, population_first, population_second):

        first_position = population_first.position
        second_position = population_second.position

        alpha = np.random.uniform(-0.1, 1.1, first_position.shape)

        offspring_first = np.multiply(alpha, first_position) + np.multiply(1 - alpha, second_position)
        offspring_second = np.multiply(alpha, second_position) + np.multiply(1 - alpha, first_position)

        offspring_first = np.clip(offspring_first, self._min_range_variables, self._max_range_variables)
        offspring_second = np.clip(offspring_second, self._min_range_variables, self._max_range_variables)

        return offspring_first, offspring_second


    def _apply_mutation(self, population):

        position = population.position

        number_mutants = int(np.ceil(self._mutation_rate * self._number_variables))

        indices = [int(i) for i in np.random.choice(range(self._number_variables), number_mutants, replace=False)]

        offspring = np.copy(position)

        offspring[0, indices] = position[0, indices] + 0.1 * (self._max_range_variables - self._min_range_variables) * np.random.randn(number_mutants)

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

                first_parent_index = self._binary_tournament_selection()
                second_parent_index = self._binary_tournament_selection()

                self._population_crossover[iter_crossover].position, \
                self._population_crossover[iter_crossover + 1].position = \
                self._apply_crossover(self._population_main[first_parent_index],
                                      self._population_main[second_parent_index])

                self._population_crossover[iter_crossover].cost = \
                self._cost_function(self._population_crossover[iter_crossover].position)

                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_function(self._population_crossover[iter_crossover + 1].position)

            for iter_mutation in range(self._number_mutation):

                parent_index = self._binary_tournament_selection()

                self._population_mutation[iter_mutation].position = \
                self._apply_mutation(self._population_main[parent_index])

                self._population_mutation[iter_mutation].cost = \
                self._cost_function(self._population_mutation[iter_mutation].position)

            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)

            self._population_main, self._pareto_front = self._non_dominated_sorting(self._population_main)

            self._population_main = self._crowding_distance(self._population_main, self._pareto_front)

            self._population_main = self._sort(self._population_main, self._pareto_front)

            self._population_main = self._population_main[:self._number_population]
            print(len(self._pareto_front[0]))
        self._population_main, self._pareto_front = self._non_dominated_sorting(self._population_main)

        os.makedirs("./figures", exist_ok=True)
        pop_cost_first_front = np.hstack([self._population_main[i].cost for i in self._pareto_front[0]])
        plt.figure(dpi=300, figsize=(10, 6))
        plt.scatter(pop_cost_first_front[0, :],
                    pop_cost_first_front[1, :], marker="x", color="red", s=5)
        plt.xlabel("first objective function")
        plt.ylabel("second objective function")
        plt.title("Non Dominated Sorting Genetic Algorithm - II", fontweight="bold")
        plt.savefig("./figures/pareto_front.png")
        toc = time.time()

        return self._population_main, self._pareto_front, toc - tic
