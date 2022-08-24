import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class PESAII:

    class _Individual:

        def __init__(self):

            self.position = None
            self.cost = None
            self.is_dominated = None
            self.grid_index = None
            self.gird_sub_index = None

    def __init__(self,
                 number_variables,
                 min_range_variables,
                 max_range_variables,
                 cost_function,
                 max_iteration,
                 number_population,
                 crossover_percentage,
                 mutation_rate,
                 number_archive,
                 number_grids,
                 selection_pressure,
                 deletion_pressure):

        self._number_variables = number_variables
        self._min_range_variables = min_range_variables
        self._max_range_variables = max_range_variables
        self._cost_fn = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_population
        self._number_archive = number_archive
        self._crossover_percentage = crossover_percentage
        self._number_grids = number_grids
        self._mutation_rate = mutation_rate
        self._population_main = [self._Individual() for _ in range(self._number_population)]
        self._number_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_population / 2))
        self._number_mutation = self._number_population - self._number_crossover
        self._population_crossover = None
        self._population_mutation = None
        self._archive = []
        self._grids = None
        self._selection_pressure = selection_pressure
        self._deletion_pressure = deletion_pressure


    def _initialize_population(self):

        for i in range(self._number_population):

            self._population_main[i].position = np.random.uniform(self._min_range_variables,
                                                                  self._max_range_variables,
                                                                  (1, self._number_variables))

            self._population_main[i].cost = self._cost_fn(self._population_main[i].position)

    @staticmethod
    def _dominate(population1_cost, population2_cost):

        return np.all(population1_cost <= population2_cost) and np.any(population1_cost < population2_cost)

    def _determine_domination(self, population):

        for i in range(len(population)):

            population[i].is_dominated = 0

        for i in range(len(population)):

            for j in range(i + 1, len(population)):

                if self._dominate(population[i].cost, population[j].cost):

                    population[j].is_dominated = 1

                elif self._dominate(population[j].cost, population[i].cost):

                    population[i].is_dominated = 1

        return population

    def _add_to_archive(self):

        for i in range(self._number_population):

            if self._population_main[i].is_dominated == 0:

                self._archive.append(copy.deepcopy(self._population_main[i]))

    def _get_grids(self):

        number_obj = self._archive[0].cost.shape[0]
        self._grids = np.zeros((number_obj, self._number_grids + 2))
        self._grids[:, -1] = np.inf

        costs = np.hstack([pop.cost for pop in self._archive])
        costs_min = np.min(costs, axis=1, keepdims=True)
        costs_max = np.max(costs, axis=1, keepdims=True)

        self._grids[:, :-1] = np.linspace(costs_min, costs_max, self._number_grids + 1).T

    def _calc_indices(self):

        for pop in self._archive:

            pos = np.zeros((pop.cost.shape[0], 1))

            for i in range(int(pop.cost.shape[0])):

                pos[i, 0] = int(np.argwhere(pop.cost[i, 0] <= self._grids[i, :])[0][0])

            pop.grid_sub_index = pos
            pop.grid_index = (self._number_grids + 2) * pos[0, 0] + pos[1, 0]

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

        mutated_cells = [int(i) for i in np.random.choice(range(self._number_variables), number_mutants, replace=False)]

        offspring = np.copy(position)

        offspring[:, mutated_cells] = position[:, mutated_cells] + (0.1 *
                                                                    (self._max_range_variables -
                                                                     self._min_range_variables) *
                                                                    np.random.randn(number_mutants))

        offspring = np.clip(offspring, self._min_range_variables, self._max_range_variables)

        return offspring

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])

    def _select_population(self):

        occupied_cells = set([pop.grid_index for pop in self._archive])
        pop_per_cells = {i: [pop for pop in self._archive if pop.grid_index == i] for i in occupied_cells}
        number_population = {i: len(pop_per_cells[i]) for i in pop_per_cells}
        probs_per_cell = {i: np.exp(- self._selection_pressure * number_population[i]) for i in pop_per_cells}
        probs_sum = sum(list(probs_per_cell.values()))
        probs_per_cell = {i: probs_per_cell[i] / probs_sum for i in pop_per_cells}

        selected_cell_index = self._roulette_wheel_selection(list(probs_per_cell.values()))
        selected_cell = list(probs_per_cell.keys())[selected_cell_index]

        return np.random.choice(pop_per_cells[selected_cell])

    def _delete_extra(self):

        while len(self._archive) > self._number_archive:

            occupied_cells = set([pop.grid_index for pop in self._archive])
            pop_per_cell = {i: [pop for pop in self._archive if pop.grid_index == i] for i in occupied_cells}
            number_pop_per_cell = {i: len(pop_per_cell[i]) for i in pop_per_cell}
            probs_per_cell = {i: np.exp(self._deletion_pressure * number_pop_per_cell[i]) for i in number_pop_per_cell}
            probs_sum = sum(list(probs_per_cell.values()))
            probs_per_cell = {i: probs_per_cell[i] / probs_sum for i in probs_per_cell}

            selected_cell_index = self._roulette_wheel_selection(list(probs_per_cell.values()))
            selected_cell = list(probs_per_cell.keys())[selected_cell_index]
            pop_to_remove = np.random.choice(pop_per_cell[selected_cell])

            self._archive.remove(pop_to_remove)

    def run(self):

        tic = time.time()

        self._initialize_population()

        self._population_main = self._determine_domination(self._population_main)

        self._add_to_archive()

        self._get_grids()

        self._calc_indices()

        for iter_main in range(self._max_iteration):
            print(iter_main)
            self._population_crossover = [self._Individual() for _ in range(self._number_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._number_mutation)]

            for iter_c in range(0, self._number_crossover, 2):

                parent1 = self._select_population()
                parent2 = self._select_population()

                self._population_crossover[iter_c].position, \
                self._population_crossover[iter_c + 1].position = self._apply_crossover(parent1, parent2)

                self._population_crossover[iter_c].cost = self._cost_fn(self._population_crossover[iter_c].position)
                self._population_crossover[iter_c+1].cost = self._cost_fn(self._population_crossover[iter_c+1].position)

            for iter_m in range(self._number_mutation):

                parent = self._select_population()

                self._population_mutation[iter_m].position = self._apply_mutation(parent)
                self._population_mutation[iter_m].cost = self._cost_fn(self._population_mutation[iter_m].position)

            self._population_main = []
            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)

            self._population_main = self._determine_domination(self._population_main)

            self._add_to_archive()

            self._archive = self._determine_domination(self._archive)

            self._archive = [pop for pop in self._archive if not pop.is_dominated]

            self._get_grids()

            self._calc_indices()
            print("\tpareto length: ", len(self._archive))
            self._delete_extra()
            print("\tpareto length: ", len(self._archive))

        toc = time.time()

        os.makedirs("./figures", exist_ok=True)
        costs = np.hstack([pop.cost for pop in self._archive])
        plt.figure(dpi=300, figsize=(10, 6))
        plt.scatter(costs[0, :], costs[1, :], marker="x", c="r", s=8)
        plt.xlabel("First Objective Function")
        plt.ylabel("Second Objective Function")
        plt.title("MOP2 Using Pareto Envelope based Selection Algorithm - II", fontweight="bold")
        plt.savefig("./figures/pareto_front.png")

        return self._population_main, self._archive, self._grids, toc - tic
