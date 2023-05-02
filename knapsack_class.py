import numpy as np


class Knapsack:
    def __init__(self, n_items, weights, values, capacity):
        self.W = weights
        self.P = values
        self.C = capacity
        self.n_var = n_items

    def show_the_problem(self):
        print(f"Number of items : {self.n_var}")
        print(f"List of weights : {self.W}")
        print(f"List of prices : {self.P}")
        print(f"Capacity : {self.C}")


def random_problem(num_of_items, seed=1):
    np.random.seed(seed)
    W = np.random.randint(1, 100, size=num_of_items)
    P = np.random.randint(1, 100, size=num_of_items)
    C = sum(W) // 5

    return Knapsack(num_of_items, W, P, C)


a = random_problem(5, seed=1)
a.show_the_problem()
