from multi_objective_optimization.mop2_using_nsgaII.cost_function import mop2
from multi_objective_optimization.nsgaII import NSGAII


def main():

    # cost function
    cost_func = mop2

    # solution method
    solution_method = NSGAII(cost_function=cost_func,
                             number_of_variables=3,
                             min_range_of_variables=-4,
                             max_range_of_variables=4,
                             max_iteration=100,
                             number_of_population=50,
                             crossover_percentage=0.8,
                             mutation_percentage=0.4,
                             mutation_rate=0.03,
                             )

    population, pareto_front = solution_method.run()

    return population, pareto_front


if __name__ == "__main__":

    pop, pareto_front = main()