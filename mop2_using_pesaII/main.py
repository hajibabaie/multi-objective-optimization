from multi_objective_optimization.mop2_using_pesaII.mop2 import cost_function
from multi_objective_optimization.mop2_using_pesaII.pesaII import PESAII


def main():


    cost_fn = cost_function

    solution_method = PESAII(number_variables=3,
                             min_range_variables=-4,
                             max_range_variables=+4,
                             cost_function=cost_fn,
                             max_iteration=100,
                             number_population=100,
                             crossover_percentage=0.8,
                             mutation_rate=0.03,
                             number_archive=100,
                             number_grids=10,
                             selection_pressure=1,
                             deletion_pressure=1)

    population_main, archive, grids, run_time = solution_method.run()

    return population_main, archive, grids, run_time


if __name__ == "__main__":

    population, repo, grid, runtime = main()
