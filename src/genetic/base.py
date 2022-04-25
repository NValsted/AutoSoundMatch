from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Optional
from warnings import warn

import numpy as np
import torch
from deap import algorithms, base, tools
from tqdm import tqdm

from src.daw.audio_model import AudioBridgeTable


@dataclass
class SimplifiedIndividual:
    fitness: list[float]
    parameters: list[float]

    @classmethod
    def from_individual(cls, individual) -> "SimplifiedIndividual":
        return dict(
            fitness=individual.fitness.getValues(),
            parameters=[param for param in individual],
        )


class NSGA2:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II

    Implementation references:
    - https://deap.readthedocs.io/en/master/examples/nsga3.html
    - https://www.human-competitive.org/sites/default/files/tatar-paper.pdf
    """

    toolbox: base.Toolbox
    evaluation_func: Callable
    out_dim: int = 32
    population_size: int = 500
    max_generations: int = 3000
    crossover_probability: float = 1.0
    mutation_probability: float = 1.0

    def __init__(
        self,
        toolbox: base.Toolbox,
        evaluation_func: Callable,
        out_dim: int = 32,
        population_size: int = 500,
        max_generations: int = 3000,
        crossover_probability: float = 1.0,
        mutation_probability: float = 1.0,
    ):
        self.toolbox = toolbox
        self.evaluation_func = evaluation_func
        self.out_dim = out_dim
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

    def _eval_fitness(self, pop):
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return invalid_ind

    def __call__(
        self,
        audio_bridge: AudioBridgeTable,
        time_limit: Optional[int] = None,
        patience: int = 200,
        verbose: bool = False,
    ):
        signal = torch.load(audio_bridge.audio_path)
        midi_path = audio_bridge.midi_path

        datetime_start = datetime.utcnow()

        self.toolbox.register(
            "evaluate",
            partial(self.evaluation_func, target_signal=signal, midi_path=midi_path),
        )

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = self.toolbox.population(n=self.population_size)
        invalid_ind = self._eval_fitness(pop)

        no_improvement_count = 0
        previous_record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **previous_record)
        if verbose:
            print(logbook.stream)

        for gen in tqdm(range(1, self.max_generations)):
            offspring = algorithms.varAnd(
                pop,
                self.toolbox,
                self.crossover_probability,
                self.mutation_probability,
            )
            invalid_ind = self._eval_fitness(offspring)

            pop = self.toolbox.select(pop + offspring, self.population_size)

            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            if sum(record["min"]) + 1e-6 < sum(previous_record["min"]):
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count > patience:
                break

            previous_record = record

            if time_limit is not None:
                if (datetime.utcnow() - datetime_start) > timedelta(minutes=time_limit):
                    warn(f"Time limit reached at {datetime.utcnow()}")
                    break

        return pop, logbook
