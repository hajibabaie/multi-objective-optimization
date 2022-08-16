import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os


class NSGAII:

    class _Individual:

        def __init__(self):

            self.position = None
            self.cost = None
            self.rank = None
            self.dominated_count = 0
            self.domination_set = []
            self.crowding_distance = None


    def __init__(self,
                 cost_function,
                 number_of_variables,
                 min_range_of_variables,
                 max_range_of_variables,
                 max_iteration,
                 number_of_population,
                 crossover_percentage,
                 mutation_percentage,
                 mutation_rate):

        self._cost_function = cost_function
        self._number_of_variables = number_of_variables
        self._min_range_of_variables = min_range_of_variables
        self._max_range_of_variables = max_range_of_variables
        self._max_iteration = max_iteration
        self._number_of_population = number_of_population
        self._population_main = None
        self._crossover_percentage = crossover_percentage
        self._number_of_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_of_population / 2))
        self._mutation_percentage = mutation_percentage
        self._number_of_mutation = int(np.ceil(self._mutation_percentage * self._number_of_population))
        self._mutation_rate = mutation_rate
        self._best_cost = []
        self._pareto_front = None
        self._population_crossover = None
        self._population_mutation = None

    def _initialize_population(self):

        population = [self._Individual() for _ in range(self._number_of_population)]

        for i in range(self._number_of_population):

            population[i].position = np.random.uniform(self._min_range_of_variables,
                                                       self._max_range_of_variables,
                                                       (1, self._number_of_variables))

            population[i].cost = self._cost_function(population[i].position)

        return population

    @staticmethod
    def _non_dominated_sorting(population):

        def dominate(population_first, population_second):

            cost_first = population_first.cost
            cost_second = population_second.cost

            return np.all(cost_first <= cost_second) and np.any(cost_first < cost_second)

        for i in range(len(population)):

            population[i].domination_set = []
            population[i].dominated_count = 0

        pareto_front = {0: []}

        for i in range(len(population)):

            for j in range(i + 1, len(population)):

                if dominate(population[i], population[j]):

                    population[i].domination_set.append(j)
                    population[j].dominated_count += 1

                elif dominate(population[j], population[i]):

                    population[j].domination_set.append(i)
                    population[i].dominated_count += 1

            if not population[i].dominated_count:

                pareto_front[0].append(i)
                population[i].rank = 0

        counter = 0
        while True:

            Q = []

            for i in pareto_front[counter]:

                for j in population[i].domination_set:

                    population[j].dominated_count -= 1

                    if not population[j].dominated_count:

                        Q.append(j)
                        population[j].rank = counter + 1

            if not len(Q):

                break
            pareto_front[counter + 1] = Q
            counter += 1

        return pareto_front, population

    @staticmethod
    def _crowding_distance(population, pareto_front):

        for front in pareto_front:

            if len(pareto_front[front]) <= 2:

                for pop_index in pareto_front[front]:

                    population[pop_index].crowding_distance = np.random.uniform(5, 10)

            else:

                sorted_index = [pareto_front[front][int(j)] for j in
                                np.argsort([population[i].cost[0] for i in pareto_front[front]])]


                first_case = population[sorted_index[0]].cost
                last_case = population[sorted_index[-1]].cost
                max_d1 = np.abs(first_case[0] - last_case[0])
                max_d2 = np.abs(first_case[1] - last_case[1])

                for i in range(len(sorted_index)):

                    if i == 0 or i == len(sorted_index) - 1:

                        population[sorted_index[i]].crowding_distance = np.random.uniform(5, 10)

                    else:

                        next_case = population[sorted_index[i + 1]].cost
                        prev_case = population[sorted_index[i - 1]].cost

                        d1 = np.abs(next_case[0] - prev_case[0]) / max_d1
                        d2 = np.abs(next_case[1] - prev_case[1]) / max_d2

                        population[sorted_index[i]].crowding_distance = d1 + d2

        return population

    @staticmethod
    def _sort(population, pareto_front):

        population_sorted = []
        for front in pareto_front:

            current_pop_argsort = np.argsort([population[i].crowding_distance for i in pareto_front[front]])[::-1]
            for j in current_pop_argsort:
                population_sorted.append(population[pareto_front[front][int(j)]])
        return population_sorted

    def _binary_tournament_selection(self):

        indices = [int(i) for i in np.random.choice(range(self._number_of_population), 2, replace=False)]

        if self._population_main[indices[0]].rank < self._population_main[indices[1]].rank:

            return indices[0]

        elif self._population_main[indices[0]].rank > self._population_main[indices[1]].rank:

            return indices[1]

        else:

            if self._population_main[indices[0]].crowding_distance > self._population_main[1].crowding_distance:

                return indices[0]

            elif self._population_main[indices[1]].crowding_distance > self._population_main[0].crowding_distance:

                return indices[1]

            else:

                return indices[1]

    def _apply_crossover(self, population_first, population_second):

        alpha = np.random.uniform(-0.1, 1.1, population_second.shape)
        offspring_first = np.multiply(alpha, population_first) + np.multiply(1 - alpha, population_second)
        offspring_second = np.multiply(alpha, population_second) + np.multiply(1 - alpha, population_first)

        offspring_first = np.clip(offspring_first, self._min_range_of_variables, self._max_range_of_variables)
        offspring_second = np.clip(offspring_second, self._min_range_of_variables, self._max_range_of_variables)

        return offspring_first, offspring_second

    def _apply_mutation(self, population):

        offspring = np.copy(population)
        index = int(np.random.choice(range(self._number_of_variables)))
        offspring[0, index] = offspring[0, index] + 0.1 * np.random.randn()

        offspring = np.clip(offspring, self._min_range_of_variables, self._max_range_of_variables)

        return offspring

    def run(self):

        self._population_main = self._initialize_population()

        self._pareto_front, self._population_main = self._non_dominated_sorting(self._population_main)
        self._population_main = self._crowding_distance(self._population_main, self._pareto_front)
        self._population_main = self._sort(self._population_main, self._pareto_front)

        for iter_main in range(self._max_iteration):

            self._population_crossover = [self._Individual() for _ in range(self._number_of_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._number_of_mutation)]

            for iter_crossover in range(0, self._number_of_crossover, 2):

                parent_first = self._binary_tournament_selection()
                parent_second = self._binary_tournament_selection()

                self._population_crossover[iter_crossover].position, \
                self._population_crossover[iter_crossover + 1].position = self._apply_crossover(self._population_main[parent_first].position,
                                                                                                self._population_main[parent_second].position)

                self._population_crossover[iter_crossover].cost = self._cost_function(self._population_crossover[iter_crossover].position)
                self._population_crossover[iter_crossover + 1].cost = self._cost_function(self._population_crossover[iter_crossover + 1].position)

            for iter_mutation in range(self._number_of_mutation):

                parent = self._binary_tournament_selection()

                self._population_mutation[iter_mutation].position = self._apply_mutation(self._population_main[parent].position)
                self._population_mutation[iter_mutation].cost = self._cost_function(self._population_mutation[iter_mutation].position)

            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)

            self._pareto_front, self._population_main = self._non_dominated_sorting(self._population_main)
            self._population_main = self._crowding_distance(self._population_main, self._pareto_front)
            self._population_main = self._sort(self._population_main, self._pareto_front)
            self._population_main = self._population_main[:self._number_of_population]
            self._pareto_front, self._population_main = self._non_dominated_sorting(self._population_main)
            self._population_main = self._crowding_distance(self._population_main, self._pareto_front)
            self._population_main = self._sort(self._population_main, self._pareto_front)

        os.makedirs("./figures", exist_ok=True)
        plt.figure(dpi=300, figsize=(10, 6))
        pop_cost = [self._population_main[i].cost for i in self._pareto_front[0]]
        plt.scatter([pop_cost[i][0] for i in range(len(pop_cost))],
                    [pop_cost[i][1] for i in range(len(pop_cost))], marker="x", c="r")
        plt.xlabel("First Objective Function")
        plt.ylabel("Second Objective Function")
        plt.title("MOP2 Using Non Dominated Sorting Genetic Algorithm - II", fontweight="bold")
        plt.savefig("./figures/nsgaII.png")

        return self._population_main, self._pareto_front
