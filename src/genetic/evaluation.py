from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import dill
from deap.tools import Logbook
from scipy.io import wavfile

from src.config.base import REGISTRY
from src.database.dataset import load_formatted_audio
from src.daw.audio_model import AudioBridgeTable
from src.daw.factory import SynthHostFactory
from src.daw.signal_processing import spectral_convergence, spectral_mse
from src.genetic.base import SimplifiedIndividual
from src.genetic.factory import NSGA2Factory
from src.genetic.pareto_front import pareto_front
from src.utils.loss_model import LossTable, TrainValTestEnum


def save_run(
    bridge: AudioBridgeTable, population: list[SimplifiedIndividual], logbook: Logbook
):
    stem = Path(bridge.audio_path).stem
    population_path = (REGISTRY.PATH.genetic / "_").with_name(f"{stem}_population.pkl")
    logbook_path = (REGISTRY.PATH.genetic / "_").with_name(f"{stem}_logbook.pkl")

    with population_path.open("wb") as f:
        dill.dump(population, f)
    with logbook_path.open("wb") as f:
        dill.dump(logbook, f)
    REGISTRY.add_blob(population_path)
    REGISTRY.add_blob(logbook_path)


def evaluate_population(
    bridge: AudioBridgeTable,
    simplified_population: list[SimplifiedIndividual],
    write_audio: bool = False,
    monophonic: bool = False,
) -> list[LossTable]:
    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))
    synth_host = sh_factory()

    if monophonic:
        raise NotImplementedError

    _, target_signal = load_formatted_audio(bridge.audio_path)
    pareto_optimal_solutions = pareto_front(simplified_population)

    if write_audio:
        file_path = bridge.audio_path.replace(".pt", ".wav")
        wavfile.write(
            file_path,
            REGISTRY.SYNTH.sample_rate,
            target_signal.cpu().numpy(),
        )
        REGISTRY.add_blob(file_path)

    evaluations = defaultdict(list)

    for i, solution in enumerate(pareto_optimal_solutions):
        synth_host.set_patch(solution["parameters"])
        inferred_audio = synth_host.render(bridge.midi_path)

        if write_audio:
            file_path = bridge.audio_path.replace(".pt", f"_inferred_{i}.wav")
            wavfile.write(
                file_path,
                REGISTRY.SYNTH.sample_rate,
                inferred_audio,
            )
            REGISTRY.add_blob(file_path)

        for loss_callable in (spectral_convergence, spectral_mse):
            loss = loss_callable(inferred_audio, target_signal)
            evaluations[str(loss_callable.__name__)].append(loss)

    losses = []
    for loss_name, loss_values in evaluations.items():
        loss_model = LossTable(
            model_id="NSGA-II",
            type=loss_name,
            train_val_test=TrainValTestEnum.TEST,
            value=sum(loss_values) / len(loss_values),
        )

        losses.append(loss_model)

    return losses


def exec_run(
    bridge: AudioBridgeTable, write_audio: bool = False, monophonic: bool = False
) -> tuple[LossTable, LossTable]:
    nsga2_factory = NSGA2Factory(
        out_dim=REGISTRY.GENETIC.out_dim,
        population_size=REGISTRY.GENETIC.population_size,
        max_generations=REGISTRY.GENETIC.max_generations,
        crossover_probability=REGISTRY.GENETIC.crossover_probability,
        mutation_probability=REGISTRY.GENETIC.mutation_probability,
    )

    nsga2 = nsga2_factory()
    population, logbook = nsga2(
        audio_bridge=bridge,
        time_limit=REGISTRY.GENETIC.time_limit,
        patience=REGISTRY.GENETIC.patience,
    )
    simplified_population = [
        SimplifiedIndividual.from_individual(ind) for ind in population
    ]

    save_run(bridge, simplified_population, logbook)

    losses = evaluate_population(
        bridge=bridge,
        simplified_population=simplified_population,
        write_audio=write_audio,
        monophonic=monophonic,
    )
    return losses


def evaluate_ga(
    audio_bridges: list[AudioBridgeTable],
    test_limit: Optional[int] = None,
    write_audio: bool = False,
    monophonic: bool = False,
) -> list[LossTable]:

    if test_limit is not None:
        audio_bridges = audio_bridges[:test_limit]

    with Pool() as p:
        loss_pairs = p.map(
            partial(exec_run, write_audio=write_audio, monophonic=monophonic),
            audio_bridges,
        )

    losses = [loss for loss_pair in loss_pairs for loss in loss_pair]

    return losses
