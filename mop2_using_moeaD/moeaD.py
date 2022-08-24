from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os



class MOEAD:

    class _SubProblem:

        def __init__(self):

            self.lambd = None
            self.neighbors = None

    class _Individual:

        def __init__(self):

            self.position = None
            self.cost = None
            self.decomposed_cost = None
            self.is_dominated = None

    def __init__(self,
                 number_variables,
                 min_range_variables,
                 max_range_variables,
                 cost_function,
                 max_iteration,
                 number_population,
                 number_objective_function,
                 number_archive):

        self._number_variables = number_variables
        self._min_range_variables = min_range_variables
        self._max_range_variables = max_range_variables
        self._cost_fn = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_population
        self._number_obj = number_objective_function
        self._number_archive = number_archive
        self._population_main = [self._Individual() for _ in range(self._number_population)]
        self._population_new = None
        self._sub_problems = [self._SubProblem() for _ in range(self._number_population)]
        self._archive = []
        self._t = 5
        self._goal = 0.015 * np.ones((1, self._number_obj))

    def _decomposed_cost(self, individual, subproblem):

        cost = individual.cost
        lambd = subproblem.lambd

        return float(np.linalg.norm(np.multiply(lambd, np.abs(cost - self._goal))))

    def _initialize_sub_problems(self):

        for i in range(self._number_population):

            lambd = np.random.rand(1, self._number_obj)
            lambd = np.divide(lambd, np.linalg.norm(lambd))
            self._sub_problems[i].lambd = lambd

        lambdas = np.vstack([sub_problem.lambd for sub_problem in self._sub_problems])
        distances = np.argsort(cdist(lambdas, lambdas, "seuclidean"), axis=0)

        for i in range(self._number_population):
            self._sub_problems[i].neighbors = [int(i) for i in distances[:, i]][:self._t]

    def _initialize_population(self):

        for i in range(self._number_population):

            self._population_main[i].position = np.random.uniform(self._min_range_variables,
                                                                  self._max_range_variables,
                                                                  (1, self._number_variables))

            self._population_main[i].cost = self._cost_fn(self._population_main[i].position)

            self._goal = np.minimum(self._goal, self._population_main[i].cost)

        for i in range(self._number_population):

            self._population_main[i].decomposed_cost = self._decomposed_cost(self._population_main[i],
                                                                             self._sub_problems[i])

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

    def _binary_tournament_selection(self):

        indices = [int(i) for i in np.random.choice(range(self._number_population), 2, replace=False)]
        index_f, index_s = indices[0], indices[1]

        if self._population_main[index_f].decomposed_cost < self._population_main[index_s].decomposed_cost:
            return index_f
        elif self._population_main[index_s].decomposed_cost < self._population_main[index_f].decomposed_cost:
            return index_s

    def _apply_crossover(self, population1, population2):

        position1 = population1.position
        position2 = population2.position

        alpha = np.random.uniform(-0.3, 1.3, position1.shape)

        offspring = np.multiply(alpha, position1) + np.multiply(1 - alpha, position2)
        offspring = np.clip(offspring, self._min_range_variables, self._max_range_variables)

        return offspring

    def _delete_extra(self):

        while len(self._archive) > self._number_archive:

            costs = np.vstack([pop.cost for pop in self._archive])
            distances = np.sort(cdist(costs, costs, "seuclidean"), axis=0)
            k = 1
            while np.min(distances[k, :]) == np.max(distances[k, :]) and k < distances.shape[0]:
                k += 1

            index_to_delete = int(np.argmin(distances[k, :]))
            self._archive.pop(index_to_delete)

    def run(self):

        tic = time.time()

        self._initialize_sub_problems()

        self._initialize_population()

        self._population_main = self._determine_domination(self._population_main)

        self._archive = [pop for pop in self._population_main if not pop.is_dominated]

        for iter_main in range(self._max_iteration):

            for i in range(self._number_population):

                self._population_new = self._Individual()

                parent1 = self._binary_tournament_selection()
                parent2 = self._binary_tournament_selection()
                while parent1 == parent2:
                    parent2 = self._binary_tournament_selection()

                self._population_new.position = self._apply_crossover(self._population_main[parent1],
                                                                      self._population_main[parent2])

                self._population_new.cost = self._cost_fn(self._population_new.position)

                self._goal = np.minimum(self._goal, self._population_new.cost)

                for j in self._sub_problems[i].neighbors:

                    self._population_new.decomposed_cost = self._decomposed_cost(self._population_new,
                                                                                 self._sub_problems[j])

                    if self._population_new.decomposed_cost < self._population_main[i].decomposed_cost:

                        self._population_main[j] = copy.deepcopy(self._population_new)

            self._population_main = self._determine_domination(self._population_main)

            self._archive = [pop for pop in self._population_main if not pop.is_dominated]

            self._delete_extra()



        toc = time.time()

        os.makedirs("./figures", exist_ok=True)
        costs = np.vstack([pop.cost for pop in self._archive])
        plt.figure(dpi=300, figsize=(10, 6))
        plt.scatter(costs[:, 0], costs[:, 1], marker="x", c="red", s=8)
        plt.xlabel("First Objective Function")
        plt.ylabel("Second Objective Function")
        plt.title("MOP2 using Multi Objective Evolutionary Algorithm Based on Decomposition", fontweight="bold")
        plt.savefig("./figures/pareto_front.png")

        return self._population_main, self._goal, self._sub_problems, self._archive, toc - tic
