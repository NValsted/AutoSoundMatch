import numpy as np

from src.genetic.base import SimplifiedIndividual


def is_pareto_efficient(fitness: list[tuple[float, ...]]) -> list[bool]:
    # references https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python  # NOQA : E501
    costs = np.array(fitness)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient.tolist()


def pareto_front(population: list[SimplifiedIndividual]) -> list[SimplifiedIndividual]:
    pareto_optimal_solutions = is_pareto_efficient(
        [ind["fitness"] for ind in population]
    )
    return [population[i] for i, b in enumerate(pareto_optimal_solutions) if b is True]
