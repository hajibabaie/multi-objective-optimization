from multi_objective_optimization.mop2_using_mopso.mop2 import cost_function
from multi_objective_optimization.mop2_using_mopso.mopso import MOPSO


def main():

    cost_fn = cost_function

    solution_method = MOPSO(number_variables=3,
                            min_range_variables=-4,
                            max_range_variables=4,
                            cost_function=cost_fn,
                            max_iteration=200,
                            number_particles=100,
                            repository_capacity=100,
                            selection_pressure=1,
                            deletion_pressure=1,
                            inertia_rate=1,
                            inertia_damping_rate=0.99,
                            personal_learning_rate=1,
                            global_learning_rate=2,
                            number_grids=10)

    particles, repository, grids, runtime = solution_method.run()

    return particles, repository, grids, runtime


if __name__ == "__main__":

    particles_main, repo, grid, run_time = main()
