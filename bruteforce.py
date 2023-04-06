# bruteforce methode for the knapsack problem by recurrence

import pymoo.problems.single.knapsack
import numpy as np


def bruteforce(situation, best=0, money=0, objets=[], final_lst=[0]):
    problem = pymoo.problems.single.knapsack.Knapsack(
        len(situation.W), situation.W, situation.P, situation.C
    )
    if len(problem.W) == 0 or problem.C == 0:
        return money, objets
    for i in range(problem.n_var):
        c = problem.C
        w = problem.W.copy()
        p = problem.P.copy()
        m = money
        obj = objets.copy()
        if w[0] <= c:
            c -= w[0]
            m += p[0]
            obj.append(1)
        else:
            obj.append(0)

        w = np.delete(w, 0)
        p = np.delete(p, 0)

        recurence = bruteforce(
            pymoo.problems.single.knapsack.Knapsack(len(w), w, p, c),
            best,
            m,
            obj,
            final_lst,
        )
        if best < recurence[0]:
            best, final_lst = recurence

        problem.W = np.delete(problem.W, 0)
        problem.P = np.delete(problem.P, 0)

    return best, final_lst


# print(
#     bruteforce(
#         pymoo.problems.single.knapsack.Knapsack(
#             4, np.array([5, 8, 7, 3]), np.array([3, 6, 2, 4]), 10
#         )
#     )
# )
