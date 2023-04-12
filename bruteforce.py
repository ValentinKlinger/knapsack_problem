# bruteforce methode for the knapsack problem by recurrence

import pymoo.problems.single.knapsack
import numpy as np


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


# print(
#     bruteforce(
#         pymoo.problems.single.knapsack.Knapsack(
#             4, np.array([5, 8, 7, 3]), np.array([3, 6, 2, 4]), 10
#         )
#     )
# )
