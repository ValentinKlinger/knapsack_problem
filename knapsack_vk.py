# knapsack_pymoo fonction from https://pymoo.org/customization/binary.html

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
import pymoo.problems.single.knapsack
import numpy as np


def knapsack_pymoo(problem):
    algorithm = GA(
        pop_size=200,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", 100), verbose=False)
    return res


def bruteforce(situation, best=0, money=0, objets=[], final_lst=[0], len_total_plm=0):
    problem = pymoo.problems.single.knapsack.Knapsack(
        len(situation.W), situation.W, situation.P, situation.C
    )
    if len_total_plm == 0:
        len_total_plm = (
            problem.n_var
        )  # len_total_plm is the number of objets at the start
    objets = objets + [0] * (len_total_plm - len(objets))

    if len(problem.W) == 0 or problem.C == 0:
        return money, objets
    for i in range(problem.n_var):
        c = problem.C

        m = money
        obj = objets.copy()
        if problem.W[0] <= c:
            c -= problem.W[0]
            m += problem.P[0]
            obj[i + len_total_plm - problem.n_var] = 1

        problem.W = np.delete(problem.W, 0)
        problem.P = np.delete(problem.P, 0)

        recurence = bruteforce(
            pymoo.problems.single.knapsack.Knapsack(
                len(problem.W), problem.W, problem.P, c
            ),
            best,
            m,
            obj,
            final_lst,
            len_total_plm,
        )
        if best < recurence[0]:
            best, final_lst = recurence

    return best, final_lst


def greedy_algorithm(situation):
    list_of_obj_in_knapsack = [0] * situation.n_var
    knapsack_obj_total_value = 0
    c = situation.C

    ratio = []
    for i in range(situation.n_var):
        ratio.append((situation.P[i] / situation.W[i], i))

    ratio.sort(reverse=True)

    for i in range(situation.n_var):
        if situation.W[ratio[i][1]] <= c:
            knapsack_obj_total_value += situation.P[ratio[i][1]]
            list_of_obj_in_knapsack[ratio[i][1]] = 1
            c -= situation.W[ratio[i][1]]

    return knapsack_obj_total_value, list_of_obj_in_knapsack
