# From https://pymoo.org/customization/binary.html

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize



def knapsack_pymoo(problem):
    algorithm = GA(
        pop_size=200,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True)

    res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=False)
    return res
# print("Best solution found: %s" % res.X.astype(int))
# print("Function value: %s" % res.F)
# print("Constraint violation: %s" % res.CV)
