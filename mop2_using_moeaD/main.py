from multi_objective_optimization.mop2_using_moeaD.mop2 import cost_function
from multi_objective_optimization.mop2_using_moeaD.moeaD import MOEAD


def main():

    cost_fn = cost_function

    solution_method = MOEAD(number_variables=3,
                            min_range_variables=-4,
                            max_range_variables=+4,
                            cost_function=cost_fn,
                            max_iteration=200,
                            number_population=300,
                            number_objective_function=2,
                            number_archive=300)

    population_main, goal, sub_problems, archive, run_time = solution_method.run()

    return population_main, goal, sub_problems, archive, run_time


if __name__ == "__main__":

    population, goal, subproblems, estimated_pareto, runtime = main()
