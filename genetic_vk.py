import numpy as np
import random
import pymoo.problems.single.knapsack


def fitness_coefficient(individual, values, weights, capacity):
    """
    Return the fitness coefficent of a individual.
    If his wheight is superior of the capacity, fitness_coefficient = 0.
    Otherwise fitness_coefficient = total value of the objects
    """
    if sum([weights[i] for i in range(len(weights)) if individual[i] == 1]) > capacity:
        return 0, individual
    return (
        sum([values[i] for i in range(len(weights)) if individual[i] == 1]),
        individual,
    )


def two_point_crossover(parents, mutation_rate):
    """
    Retourn 2 individuals who are the two point crossover of their parents
    """
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
    if random.random() < mutation_rate:
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
            ),
            mutation_rate,
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


def mutation(parents, mutation_rate):
    """
    Return the parents with a random genetic modification
    """
    children = parents.copy()

    for child in children:
        chromosome_switch_index = random.randint(0, len(child) - 1)

        if child[chromosome_switch_index] == 0:
            child[chromosome_switch_index] = 1

        else:
            child[chromosome_switch_index] = 0

    if random.random() < mutation_rate:
        return mutation(children, mutation_rate)
    return children


def natural_selection(population, NUM_FIGHTERS):
    """
    Return a array of NUM_FIGHTERS individuals from population
    """
    fighters_location = np.random.choice(
        population.shape[0], size=NUM_FIGHTERS, replace=False
    )

    fighters = population[fighters_location, :]
    return fighters


def tournament(NUM_FIGHTERS, fighters, values, weights, capacity):
    """
    Find a mate of parents by a tournament
    """
    if NUM_FIGHTERS == 2:
        return fighters

    winers = None
    for i in range(0, int(NUM_FIGHTERS), 2):
        if (
            fitness_coefficient(fighters[i], values, weights, capacity)[0]
            > fitness_coefficient(fighters[i + 1], values, weights, capacity)[0]
        ):
            winer = fighters[i]
        else:
            winer = fighters[i + 1]

        if winers is None:
            winers = np.array(winer)
        else:
            winers = np.vstack((winers, winer))

    return tournament(NUM_FIGHTERS / 2, winers, values, weights, capacity)


def genetic_algorithm(problem):
    """
    Genetic_algorithm V2.
    In this version, I incrase the number of participants in the turnament.
    I used two pairs, one created by tournaments and
    the other created by the best individual and the winner of a tournament.
    """
    # Population with N individuals.
    N = 250  # Have to be even.
    NUM_OF_GENERATION = 500

    # Creation of the first population.
    average_weights = sum(problem.W)
    capacity_on_arevage = problem.C / average_weights

    population = np.zeros((N, problem.n_var), dtype=int)
    for individual in range(N):
        for chromosome in range(problem.n_var):
            if random.random() < capacity_on_arevage:
                population[individual, chromosome] = 1

    # Genetic process.
    NUM_FIGHTERS = 32
    NUM_FIGHTERS_2 = 32

    for i in range(NUM_OF_GENERATION):
        fighters = natural_selection(population, NUM_FIGHTERS + NUM_FIGHTERS_2)

        best_solution = (0, np.zeros((problem.n_var), dtype=int))
        for individual in population:
            if (
                fitness_coefficient(individual, problem.P, problem.W, problem.C)[0]
                > best_solution[0]
            ):
                best_solution = fitness_coefficient(
                    individual, problem.P, problem.W, problem.C
                )
        # Parents 1

        parents_1 = tournament(
            NUM_FIGHTERS, fighters[:32], problem.P, problem.W, problem.C
        )

        # Parents 2
        parents_2_tournament_results = tournament(
            NUM_FIGHTERS_2, fighters[32:], problem.P, problem.W, problem.C
        )
        if (
            fitness_coefficient(
                parents_2_tournament_results[0], problem.P, problem.W, problem.C
            )[0]
            > fitness_coefficient(
                parents_2_tournament_results[1], problem.P, problem.W, problem.C
            )[0]
        ):
            winer = parents_2_tournament_results[0]
        else:
            winer = parents_2_tournament_results[1]

        parents_2 = np.vstack((best_solution[1], winer))

        population = None
        CROSSOVER_RATE = 0.6
        REPRODUCTION_RATE = 0.385
        MUTATION_RATE = 0.015
        while population is None or len(population) < N:
            random_action = random.random()

            if random.randint(0, 1) == 0:
                parents = parents_1
            else:
                parents = parents_2

            if random_action < CROSSOVER_RATE:
                if population is None:
                    population = two_point_crossover(parents, MUTATION_RATE)
                else:
                    population = np.vstack(
                        (population, two_point_crossover(parents, MUTATION_RATE))
                    )

            elif random_action < CROSSOVER_RATE + REPRODUCTION_RATE:
                if population is None:
                    population = parents
                else:
                    population = np.vstack((population, parents))

            else:
                if population is None:
                    population = mutation(parents, MUTATION_RATE)
                else:
                    population = np.vstack(
                        (population, mutation(parents, MUTATION_RATE))
                    )

    # Find the best of the latest generation.
    out = (0, [0] * problem.n_var)
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


# problem = pymoo.problems.single.knapsack.create_random_knapsack_problem(25)
# print(genetic_algorithm(problem))
