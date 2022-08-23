from multi_objective_optimization.mop2_using_speaII.mop2 import cost_function
from multi_objective_optimization.mop2_using_speaII.speaII import SPEAII


def main():

    cost_fn = cost_function

    solution_method = SPEAII(number_variables=3,
                             min_range_variables=-4,
                             max_range_variables=4,
                             cost_function=cost_fn,
                             max_iteration=100,
                             number_population=100,
                             number_archive=100,
                             crossover_percentage=0.8,
                             mutation_rate=0.03)

    population_main, population_total, archive, run_time = solution_method.run()

    return population_main, population_total, archive, run_time


if __name__ == "__main__":

    population, total, repository, runtime = main()
