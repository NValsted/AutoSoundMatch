import random
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Callable

import torch
import torch.fft as FFT
from deap import base, creator, tools
from scipy.signal import hilbert
from torchaudio.functional import lowpass_biquad

from src.config.base import REGISTRY
from src.daw.factory import SynthHostFactory
from src.daw.signal_transformers import StereoToMono
from src.genetic.base import NSGA2
from src.utils.temporary_context import temporary_attrs

SH_FACTORY = SynthHostFactory(**dict(REGISTRY.SYNTH))
SYNTH_HOST = SH_FACTORY()
STEREO_TO_MONO = StereoToMono((None, 2))

OBJECTIVES = [
    (FFT.rfft, torch.abs),
    (partial(torch.stft, n_fft=1024, hop_length=512, return_complex=True), torch.abs),
    (
        hilbert,
        torch.from_numpy,
        torch.abs,
        partial(lowpass_biquad, sample_rate=REGISTRY.SYNTH.sample_rate, cutoff_freq=10),
    ),
]


def _uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def _evaluate(solution, target_signal: torch.Tensor, midi_path: Path) -> list[float]:
    SYNTH_HOST.set_patch(solution)
    inferred_audio = torch.from_numpy(SYNTH_HOST.render(str(midi_path)))

    fitness = []
    for objective in OBJECTIVES:

        def _func(x):
            return reduce(lambda acc, f: f(acc), objective, x)

        diff = _func(STEREO_TO_MONO(target_signal)) - _func(
            STEREO_TO_MONO(inferred_audio)
        )
        squared_diff = torch.pow(diff, 2)
        euclidean_norm = torch.sqrt(torch.sum(squared_diff)).float()
        fitness.append(euclidean_norm.item())

    return fitness


@dataclass
class NSGA2Factory:
    attr_float_func: Callable = _uniform
    out_dim: int = 32
    bound_low: float = 0.0
    bound_up: float = 1.0
    population_size: int = 500
    max_generations: int = 3000
    crossover_probability: float = 1.0
    mutation_probability: float = 1.0

    def __call__(self, *args, **kwargs) -> NSGA2:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            tmp: NSGA2Factory

            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register(
                "attr_float",
                tmp.attr_float_func,
                tmp.bound_low,
                tmp.bound_up,
                tmp.out_dim,
            )
            toolbox.register(
                "individual", tools.initIterate, creator.Individual, toolbox.attr_float
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register(
                "mate",
                tools.cxSimulatedBinaryBounded,
                low=tmp.bound_low,
                up=tmp.bound_up,
                eta=30.0,
            )
            toolbox.register(
                "mutate",
                tools.mutPolynomialBounded,
                low=tmp.bound_low,
                up=tmp.bound_up,
                eta=20.0,
                indpb=1.0 / tmp.out_dim,
            )
            toolbox.register("select", tools.selNSGA2)
            return NSGA2(
                toolbox=toolbox,
                evaluation_func=_evaluate,
                out_dim=tmp.out_dim,
                population_size=tmp.population_size,
                max_generations=tmp.max_generations,
                crossover_probability=tmp.crossover_probability,
                mutation_probability=tmp.mutation_probability,
            )
