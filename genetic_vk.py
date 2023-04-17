import numpy as np
import random
import pymoo.problems.single.knapsack


def fitness_coefficient(individual, values, weights, capacity):
    if sum([weights[i] for i in range(len(weights)) if individual[i] == 1]) > capacity:
        return 0, individual
    return (
        sum([values[i] for i in range(len(weights)) if individual[i] == 1]),
        individual,
    )


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


def natural_selection(population, NUM_FIGHTERS):
    # breakpoint()
    fighters_location = np.random.choice(
        population.shape[0], size=NUM_FIGHTERS, replace=False
    )

    fighters = population[fighters_location, :]
    return fighters


def tournament(NUM_FIGHTERS, fighters, values, weights, capacity):
    if NUM_FIGHTERS == 2:
        return fighters

    winers = None
    # print(NUM_FIGHTERS)
    for i in range(0, int(NUM_FIGHTERS), 2):
        # breakpoint()
        if (
            fitness_coefficient(fighters[i], values, weights, capacity)[0]
            > fitness_coefficient(fighters[i + 1], values, weights, capacity)[0]
        ):
            winer = fighters[i]
        else:
            winer = fighters[i + 1]
        # winer = max(
        #     fitness_coefficient(fighters[i], values, weights, capacity),
        #     fitness_coefficient(fighters[i + 1], values, weights, capacity),
        # )
        # breakpoint()
        if winers is None:
            winers = np.array(winer)
            # print("a")
        else:
            winers = np.vstack((winers, winer))
    # breakpoint()
    return tournament(NUM_FIGHTERS / 2, winers, values, weights, capacity)


def genetic_algorithm(problem):
    # Generation of the first population with N individuals
    N = 500

    average_weights = sum(problem.W)
    capacity_on_arevage = problem.C // average_weights

    population = np.zeros((N, problem.n_var), dtype=int)

    for individual in range(N):
        for chromosome in range(problem.n_var):
            if random.random() < capacity_on_arevage:
                population[individual, chromosome] = 1

    NUM_FIGHTERS = 8  # have to be a powers of 2 such that NUM_FIGHTERS ∈ ℕ/{0}

    for i in range(500):
        fighters = natural_selection(population, NUM_FIGHTERS)

        parents = tournament(NUM_FIGHTERS, fighters, problem.P, problem.W, problem.C)

        population = None
        while population is None or len(population) < N:
            random_action = random.random()
            if random_action < 0.6:
                if population is None:
                    population = two_point_crossover(parents)
                else:
                    population = np.vstack((population, two_point_crossover(parents)))
            elif random_action < 0.985:
                if population is None:
                    population = parents
                else:
                    population = np.vstack((population, parents))
            else:
                if population is None:
                    population = mutation(parents)
                else:
                    population = np.vstack((population, mutation(parents)))

    out = (0, [0] * problem.n_var)
    # breakpoint()
    for individual in range(N):
        if (
            fitness_coefficient(
                population[individual], problem.P, problem.W, problem.C
            )[0]
            > out[0]
        ):
            out = fitness_coefficient(
                population[individual], problem.P, problem.W, problem.C
            )

    return out


# problem = pymoo.problems.single.knapsack.Knapsack(
#     6, np.array([8, 8, 7, 3, 12, 2]), np.array([6, 40, 2, 4, 4, 1]), 8
# )
#
# print(genetic_algorithm(problem))
