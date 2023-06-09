import numpy as np
import random
import knapsack_class
from math import ceil


def fitness_coefficient(
    individual: np.ndarray[float],
    values: np.ndarray[float],
    weights: np.ndarray[float],
    capacity: np.int64,
) -> tuple[int, np.ndarray[float]]:
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
        if random.random() < mutation_rate:
            return mutation(parents, mutation_rate)
        return parents

    if len(parents[0]) == 2:
        parent1 = np.expand_dims(parents[0][0], axis=0)
        parent2 = np.expand_dims(parents[1][1], axis=0)
        parent3 = np.expand_dims(parents[1][0], axis=0)
        parent4 = np.expand_dims(parents[0][1], axis=0)
        if random.random() < mutation_rate:
            return mutation(
                np.array(
                    [
                        np.concatenate((parent1, parent2)),
                        np.concatenate((parent3, parent4)),
                    ]
                ),
                mutation_rate,
            )
        return np.array(
            [
                np.concatenate((parent1, parent2)),
                np.concatenate((parent3, parent4)),
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
    fighters_location = np.random.choice(population.shape[0], size=NUM_FIGHTERS)

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
    Genetic_algorithm V3.
    in this version, there are more parents, everyone takes part in the tournaments,
    we keep the results of the last 5 populations to find the best one.
    """

    NUM_INDIVIDUALS = 20 + 4 * problem.n_var
    NUM_OF_GENERATION = 20 + 4 * problem.n_var

    average_weights = sum(problem.W)
    capacity_on_arevage = problem.C / average_weights

    population = np.zeros((NUM_INDIVIDUALS, problem.n_var), dtype=int)
    for individual in range(NUM_INDIVIDUALS):
        for chromosome in range(problem.n_var):
            if random.random() < capacity_on_arevage:
                population[individual, chromosome] = 1

    # Genetic process.
    num_mates = ceil(NUM_INDIVIDUALS / 4)
    num_even_children_by_parents = ceil(NUM_INDIVIDUALS / (2 * num_mates))

    LEN_TOURNAMENT = 4

    NUM_INDIVIDUALS = 2 * num_mates * num_even_children_by_parents

    population_last_5g = np.empty((1, problem.n_var))
    for i in range(NUM_OF_GENERATION):
        mates = np.zeros((num_mates, 2, problem.n_var))

        for mate in range(num_mates):
            fighters = natural_selection(population, LEN_TOURNAMENT)
            mates[mate] = tournament(
                LEN_TOURNAMENT,
                fighters,
                problem.P,
                problem.W,
                problem.C,
            )

        a = 0
        population = np.zeros((NUM_INDIVIDUALS, problem.n_var))
        CROSSOVER_RATE = 0.6
        REPRODUCTION_RATE = 0.385
        MUTATION_RATE = 0.015

        counter = 0
        for parents in mates:
            for _ in range(num_even_children_by_parents):
                random_action = random.random()
                if random_action < CROSSOVER_RATE:
                    population[counter], population[counter + 1] = two_point_crossover(
                        parents, MUTATION_RATE
                    )

                elif random_action < CROSSOVER_RATE + REPRODUCTION_RATE:
                    population[counter], population[counter + 1] = parents

                else:
                    population[counter], population[counter + 1] = mutation(
                        parents, MUTATION_RATE
                    )

                counter += 2
        if i >= NUM_OF_GENERATION - 5:
            population_last_5g = np.concatenate(
                (population_last_5g, population),
            )

    # Find the best of the latest generation.
    out = (0, [0] * problem.n_var)
    for individual in range(NUM_INDIVIDUALS * 3):
        if (
            fitness_coefficient(
                population_last_5g[individual], problem.P, problem.W, problem.C
            )[0]
            > out[0]
        ):
            out = fitness_coefficient(
                population_last_5g[individual], problem.P, problem.W, problem.C
            )

    return out


problem = knapsack_class.random_problem(21)
print(genetic_algorithm(problem))
