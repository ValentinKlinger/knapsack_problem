import numpy as np
import random
import pymoo.problems.single.knapsack


def genetic_algorithm(problem):
    # Generation of the first population with N individuals
    N = 500

    random_float = random.random()


def fitness_coefficient(individual, values, weights, capacity):
    if sum([weights[i] for i in range(len(weights)) if individual[i] == 1]) > capacity:
        return 0
    return sum([values[i] for i in range(len(weights)) if individual[i] == 1])


def two_point_crossover(parents):
    if len(parents[0]) == 1:
        if random.random() < 0.015:
            return mutation(parents)
        return parents

    if len(parents[0]) == 2:
        if random.random() < 0.015:
            return mutation(
                np.array(
                    [
                        np.concatenate((parents[0][0], parents[1][1])),
                        np.concatenate((parents[1][0], parents[0][1])),
                    ]
                )
            )
        return np.array(
            [
                np.concatenate((parents[0][0], parents[1][1])),
                np.concatenate((parents[1][0], parents[0][1])),
            ]
        )
    if random.random() < 0.015:
        return mutation(
            np.array(
                [
                    np.concatenate(
                        (
                            parents[0][: len(parents[0]) // 3],
                            parents[1][
                                len(parents[0]) // 3 : 2 * (len(parents[0]) // 3)
                                + (len(parents[0]) % 3)
                            ],
                            parents[0][len(parents[0]) - len(parents[0]) // 3 :],
                        )
                    ),
                    np.concatenate(
                        (
                            parents[1][: len(parents[0]) // 3],
                            parents[0][
                                len(parents[0]) // 3 : 2 * (len(parents[0]) // 3)
                                + (len(parents[0]) % 3)
                            ],
                            parents[1][len(parents[0]) - len(parents[0]) // 3 :],
                        )
                    ),
                ]
            )
        )

    return np.array(
        [
            np.concatenate(
                (
                    parents[0][: len(parents[0]) // 3],
                    parents[1][
                        len(parents[0]) // 3 : 2 * (len(parents[0]) // 3)
                        + (len(parents[0]) % 3)
                    ],
                    parents[0][len(parents[0]) - len(parents[0]) // 3 :],
                )
            ),
            np.concatenate(
                (
                    parents[1][: len(parents[0]) // 3],
                    parents[0][
                        len(parents[0]) // 3 : 2 * (len(parents[0]) // 3)
                        + (len(parents[0]) % 3)
                    ],
                    parents[1][len(parents[0]) - len(parents[0]) // 3 :],
                )
            ),
        ]
    )


def reproduction():
    pass


def mutation(parents):
    children = parents.copy()

    for child in children:
        chromosome_switch_index = random.randint(0, len(child) - 1)

        if child[chromosome_switch_index] == 0:
            child[chromosome_switch_index] = 1

        else:
            child[chromosome_switch_index] = 0

    if random.random() < 0.015:
        return mutation(children)
    return children
