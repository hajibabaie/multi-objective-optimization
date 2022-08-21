import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class MOPSO:

    class _Particle:

        def __init__(self):
            self.position = None
            self.velocity = None
            self.cost = None
            self.position_best = None
            self.cost_best = None
            self.is_dominated = None
            self.grid_index = None
            self.grid_sub_index = None

    def __init__(self,
                 number_variables,
                 min_range_variables,
                 max_range_variables,
                 cost_function,
                 max_iteration,
                 number_particles,
                 repository_capacity,
                 selection_pressure,
                 deletion_pressure,
                 inertia_rate,
                 inertia_damping_rate,
                 personal_learning_rate,
                 global_learning_rate,
                 number_grids):

        self._number_variables = number_variables
        self._min_range_variables = min_range_variables
        self._max_range_variables = max_range_variables
        self._cost_fn = cost_function
        self._max_iteration = max_iteration
        self._number_particles = number_particles
        self._repository_cap = repository_capacity
        self._selection_pressure = selection_pressure
        self._deletion_pressure = deletion_pressure
        self._inertia_rate = inertia_rate
        self._damping_rate = inertia_damping_rate
        self._personal_rate = personal_learning_rate
        self._global_rate = global_learning_rate
        self._number_grids = number_grids
        self._repository = []
        self._grids = None
        self._particles = None

    def _initialize_particles(self):

        particles = [self._Particle() for _ in range(self._number_particles)]
        for i in range(self._number_particles):

            particles[i].position = np.random.uniform(self._min_range_variables,
                                                      self._max_range_variables,
                                                      (1, self._number_variables))

            particles[i].position_best = np.copy(particles[i].position)

            particles[i].velocity = np.zeros_like(particles[i].position)

            particles[i].cost = self._cost_fn(particles[i].position)

            particles[i].cost_best = np.copy(particles[i].cost)

        return particles

    @staticmethod
    def _dominate(particle1_cost, particle2_cost):

        return np.all(particle1_cost <= particle2_cost) and np.any(particle1_cost < particle2_cost)

    def _determine_domination(self, particles):

        for i in range(len(particles)):

            particles[i].is_dominated = 0

        for i in range(len(particles)):

            for j in range(i + 1, len(particles)):

                if self._dominate(particles[i].cost, particles[j].cost):

                    particles[j].is_dominated = 1

                elif self._dominate(particles[j].cost, particles[i].cost):

                    particles[i].is_dominated = 1

        return particles

    def _add_to_repo(self):

        for i in range(self._number_particles):

            if self._particles[i].is_dominated == 0:

                self._repository.append(copy.deepcopy(self._particles[i]))

    def _get_grids(self):

        number_obj = self._repository[0].cost.shape[0]

        grids = np.zeros((number_obj, self._number_grids + 2))
        grids[:, -1] = np.inf

        costs = np.hstack([self._repository[i].cost for i in range(len(self._repository))])
        costs_min = np.min(costs, axis=1, keepdims=True)
        costs_max = np.max(costs, axis=1, keepdims=True)

        grids[:, :-1] = np.linspace(costs_min[:, 0], costs_max[:, 0], self._number_grids + 1).T

        return grids

    def _grid_index_sub_index(self):

        for particle in self._repository:

            pos = np.zeros_like(particle.cost)

            for i in range(int(pos.shape[0])):

                pos[i, 0] = int(np.argwhere(particle.cost[i, 0] <= self._grids[i, :])[0][0])

            particle.grid_sub_index = pos
            particle.grid_index = int((self._number_grids + 2) * pos[0, 0] + pos[1, 0])

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])

    def _select_leader(self):

        occupied_cells = set([self._repository[i].grid_index for i in range(len(self._repository))])
        particle_per_cell = {i: [self._repository[j] for j in range(len(self._repository))
                                 if self._repository[j].grid_index == i]
                             for i in occupied_cells}

        number_particle_per_cell = {i: len(particle_per_cell[i]) for i in occupied_cells}

        probs_per_cell = {i: np.exp(-self._selection_pressure * number_particle_per_cell[i]) for i in occupied_cells}

        probs_sum = np.sum(list(probs_per_cell.values()))

        probs_per_cell = {i: probs_per_cell[i] / probs_sum for i in occupied_cells}

        selected_cell_index = self._roulette_wheel_selection(list(probs_per_cell.values()))

        selected_cell = list(probs_per_cell.keys())[selected_cell_index]

        leader = np.random.choice(particle_per_cell[selected_cell])

        return leader

    def _update_particles(self):

        for i in range(self._number_particles):

            leader = self._select_leader()

            self._particles[i].velocity = (self._inertia_rate * self._particles[i].velocity) + \
                                          (self._personal_rate * np.random.rand(*self._particles[i].position.shape) *
                                           (self._particles[i].position_best - self._particles[i].position)) + \
                                          (self._global_rate * np.random.rand(*self._particles[i].position.shape) *
                                           (leader.position - self._particles[i].position))

            self._particles[i].position = self._particles[i].position + self._particles[i].velocity

            self._particles[i].cost = self._cost_fn(self._particles[i].position)

            if self._dominate(self._particles[i].cost, self._particles[i].cost_best):

                self._particles[i].cost_best = np.copy(self._particles[i].cost)
                self._particles[i].position_best = np.copy(self._particles[i].position)

            elif np.random.rand() < 0.5 and not self._dominate(self._particles[i].cost_best, self._particles[i].cost):

                self._particles[i].cost_best = np.copy(self._particles[i].cost)
                self._particles[i].position_best = np.copy(self._particles[i].position)

    def _delete_extra(self):

        occupied_cells = set([self._repository[i].grid_index for i in range(len(self._repository))])
        while len(self._repository) > self._repository_cap:

            particle_per_cell = {i: [self._repository[j] for j in range(len(self._repository))
                                     if self._repository[j].grid_index == i] for i in occupied_cells}

            number_particle_per_cell = {i: len(particle_per_cell[i]) for i in occupied_cells}

            probs_per_cell = {i: np.exp(self._deletion_pressure * number_particle_per_cell[i]) for i in occupied_cells}

            probs_sum = np.sum(list(probs_per_cell.values()))

            probs_per_cell = {i: probs_per_cell[i] / probs_sum for i in occupied_cells}

            selected_cell_index = self._roulette_wheel_selection(list(probs_per_cell.values()))

            selected_cell = list(probs_per_cell.keys())[selected_cell_index]

            particle_to_remove = np.random.choice(particle_per_cell[selected_cell])

            self._repository.remove(particle_to_remove)


    def run(self):

        tic = time.time()

        self._particles = self._initialize_particles()

        self._particles = self._determine_domination(self._particles)

        self._add_to_repo()

        self._grids = self._get_grids()

        self._grid_index_sub_index()

        for iter_main in range(self._max_iteration):

            self._update_particles()

            self._particles = self._determine_domination(self._particles)

            self._add_to_repo()

            self._repository = self._determine_domination(self._repository)

            self._repository = [self._repository[i] for i in range(len(self._repository))
                                if not self._repository[i].is_dominated]

            self._grids = self._get_grids()

            self._grid_index_sub_index()

            self._delete_extra()

            self._inertia_rate *= self._damping_rate


        toc = time.time()


        os.makedirs("./figures", exist_ok=True)

        costs = np.hstack([self._repository[i].cost for i in range(len(self._repository))])

        plt.figure(dpi=300, figsize=(10, 6))
        plt.scatter(costs[0, :], costs[1, :], marker="x", c="r", s=8)
        plt.xlabel("First Objective Function")
        plt.ylabel("Second Objective Function")
        plt.title("MOP2 Using Multi Objective Particle Swarm Optimization", fontweight="bold")
        plt.savefig("./figures/pareto_front.png")

        return self._particles, self._repository, self._grids, toc - tic
