from typing import Optional

import typer

app = typer.Typer()


@app.command()
def setup_paths(
    downloads: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    audio: Optional[str] = typer.Option(None),
    midi: Optional[str] = typer.Option(None),
):

    from pathlib import Path

    from src.config.base import REGISTRY
    from src.config.registry_sections import PathSection

    path_kwargs = {}
    if downloads is not None:
        path_kwargs["downloads"] = downloads
    if model is not None:
        path_kwargs["model"] = model
    if audio is not None:
        path_kwargs["audio"] = audio
    if midi is not None:
        path_kwargs["midi"] = midi

    resolved_kwargs = {}
    for k, v in path_kwargs.items():
        resolved_path = Path(v).resolve()
        resolved_kwargs[k] = resolved_path
        if len(resolved_path.suffix) == 0:
            resolved_path.mkdir(parents=True, exist_ok=True)
        else:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

    REGISTRY.PATH = PathSection(**resolved_kwargs)
    REGISTRY.commit()


@app.command()
def register():
    """
    Interactive interface to manually register values in the registry.
    """

    import json

    from src.config.base import REGISTRY
    from src.config.registry_sections import RegistrySectionsEnum, RegistrySectionsMap

    typer.echo(f"Availble sections: {[e.value for e in RegistrySectionsEnum]}")

    sections = dict()

    while (entry := typer.prompt("Entry (leave blank to continue)", default="")) != "":
        entry = entry.split(" ")
        if len(entry) == 4:
            section, key, value, as_json = entry
        elif len(entry) == 3:
            section, key, value = entry
            as_json = False
        else:
            typer.echo(f"Invalid entry: {entry}")
            continue

        if section not in RegistrySectionsEnum.__members__:
            typer.echo(f"Invalid section: {section}")
            continue

        if as_json:
            value = json.loads(value)
        sections[section] = sections.get(section, dict())
        sections[section][key] = value

    typer.echo("Staged changes:")
    for section, section_dict in sections.items():
        typer.echo(f"{section}: {section_dict}")
    typer.confirm("Are you sure you want to register these values?", abort=True)

    for k, v in sections.items():
        sectionModel = RegistrySectionsMap[k](**v)
        REGISTRY.__setattr__(k, sectionModel)

    REGISTRY.commit()


@app.command()
def inspect_registry():
    """
    Display the current registry values.
    """
    from src.config.base import REGISTRY

    for k, v in dict(REGISTRY).items():
        typer.echo(f"{k}: {v}")
    # TODO : display info about blobs - number of blobs, combined size of blobs, etc.


@app.command()
def reset():
    """
    Reset the registry to default values, drop tables, and remove generated data.
    """
    from src.config.base import REGISTRY, Registry
    from src.database.factory import DBFactory

    if REGISTRY.DATABASE is not None:
        from src.daw.audio_model import AudioBridgeTable  # NOQA: F401
        from src.daw.synth_model import SynthParamsTable  # NOQA: F401
        from src.utils.loss_model import LossTable  # NOQA: F401

        db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)

        db = db_factory()
        db.drop_tables()

    REGISTRY.clear_blobs()
    REGISTRY = Registry()
    REGISTRY.commit()


@app.command()
def partition_midi_files(directory: list[str] = typer.Option([])):
    """
    Takes a list of directories containing midi files and partitions them into
    fixed size chunks.
    """
    from pathlib import Path
    from uuid import uuid4

    from mido import MidiFile

    from src.config.base import REGISTRY
    from src.midi.generation import partition_midi

    for dir in directory:
        file_paths = Path(dir).glob("*.mid")
        midi_files = [MidiFile(path) for path in file_paths]
        partitioned_files = partition_midi(midi_files)

        for file in partitioned_files:
            save_path = REGISTRY.PATH.midi / f"{str(uuid4())}.mid"
            save_path = save_path.resolve()
            file.save(save_path)
            REGISTRY.add_blob(save_path)


@app.command()
def setup_relational_models(
    synth_path: str = typer.Option(...), engine_url: Optional[str] = typer.Option(None)
):
    """
    Create tables in a local database for storing audio files and VST
    parameters.
    """
    from pathlib import Path

    from src.config.base import REGISTRY
    from src.config.registry_sections import SynthSection
    from src.database.factory import DBFactory
    from src.daw.factory import SynthHostFactory

    db_factory_kwargs = dict()
    if engine_url is not None:
        db_factory_kwargs["engine_url"] = engine_url
    db_factory = DBFactory(**db_factory_kwargs)

    REGISTRY.SYNTH = SynthSection(synth_path=Path(synth_path))
    sh_factory = SynthHostFactory(
        synth_path=REGISTRY.SYNTH.synth_path,
        sample_rate=REGISTRY.SYNTH.sample_rate,
        buffer_size=REGISTRY.SYNTH.buffer_size,
        duration=REGISTRY.SYNTH.duration,
        bpm=REGISTRY.SYNTH.bpm,
    )

    synth_host = sh_factory()
    definition_path = synth_host.create_parameter_table()
    typer.echo(
        f"Created model and table definition for {REGISTRY.SYNTH.synth_path} at"
        f" {definition_path}"
    )

    from src.daw.audio_model import AudioBridgeTable  # NOQA: F401
    from src.daw.synth_model import SynthParamsTable  # NOQA: F401
    from src.utils.loss_model import LossTable  # NOQA: F401

    db = db_factory()
    db.create_tables()

    db_factory.register(commit=True)


@app.command()
def generate_param_tuples(
    patches_per_midi: int = typer.Option(5),
):
    """
    Generate tuples of audio files with corresponding midi files, render
    parameters and parameters from a VST instrument.
    """
    import random

    import torch
    from librosa.util import valid_audio
    from librosa.util.exceptions import ParameterError
    from tqdm import tqdm

    from src.config.base import REGISTRY
    from src.database.factory import DBFactory
    from src.daw.audio_model import AudioBridgeTable
    from src.daw.factory import SynthHostFactory
    from src.midi.generation import generate_midi

    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))
    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)

    synth_host = sh_factory()
    db = db_factory()

    generate_midi()
    midi_paths = list(REGISTRY.PATH.midi.glob("*.mid"))
    for midi_file_path in tqdm(midi_paths[:10]):
        for i in range(patches_per_midi):
            synth_host.set_random_patch()
            audio = synth_host.render(midi_file_path)
            audio_file_path = REGISTRY.PATH.audio / (midi_file_path.name).replace(
                midi_file_path.suffix, f"_{i}.pt"
            )

            try:
                valid_audio(audio)
                audio_as_tensor = torch.from_numpy(audio).float()
                if audio_as_tensor.min() == 0 and audio_as_tensor.max() == 0:
                    raise ParameterError

            except ParameterError:
                typer.echo(f"Skipping invalid audio: {audio_file_path}")
                synth_host = sh_factory()
                continue

            torch.save(audio_as_tensor, audio_file_path)
            REGISTRY.add_blob(audio_file_path)

            synth_params = synth_host.get_patch_as_model(table=True)
            audio_bridge = AudioBridgeTable(
                audio_path=str(audio_file_path),
                midi_path=str(midi_file_path),
                synth_params=synth_params.id,
                test_flag=True if random.random() < 0.1 else False,
            )
            db.add([synth_params])
            db.add([audio_bridge])


@app.command()
def process_audio():
    from pathlib import Path

    import torch
    from sqlmodel import select
    from tqdm import tqdm

    from src.config.base import PYTORCH_DEVICE, REGISTRY
    from src.database.factory import DBFactory
    from src.daw.audio_model import AudioBridgeTable
    from src.daw.synth_model import SynthParamsTable  # NOQA: F401
    from src.utils.signal_processing import process_sample

    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()

    with db.session() as session:
        query = select(AudioBridgeTable).where()
        audio_bridges = session.exec(query).all()

    for bridge in tqdm(audio_bridges):
        signal = torch.load(bridge.audio_path).to(PYTORCH_DEVICE)
        processed = process_sample(signal)

        save_path = REGISTRY.PATH.processed_audio / Path(bridge.audio_path).name
        bridge.processed_path = str(save_path.resolve())

        torch.save(processed, save_path)
        REGISTRY.add_blob(save_path)

    db.add(audio_bridges)


@app.command()
def train_model(validation_split: Optional[float] = typer.Option(0.15)):
    """
    Train a model to estimate parameters.
    """
    from torch.utils.data import DataLoader, random_split

    from src.config.base import REGISTRY
    from src.database.dataset import FlowSynthDataset
    from src.database.factory import DBFactory
    from src.flow_synthesizer.api import get_model, prepare_registry

    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()
    dataset = FlowSynthDataset(db, shuffle=True)
    prepare_registry(dataset=dataset, commit=True)

    train_kwargs = dict(epochs=REGISTRY.TRAINMETA.epochs)

    if validation_split is not None:
        train_size = int(len(dataset) * (1 - validation_split))
        validation_size = len(dataset) - train_size

        train_dataset, validation_dataset = random_split(
            dataset, [train_size, validation_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=REGISTRY.TRAINMETA.batch_size
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=REGISTRY.TRAINMETA.batch_size
        )

        train_kwargs["train_loader"] = train_loader
        train_kwargs["validation_loader"] = validation_loader

    else:
        train_loader = DataLoader(dataset, batch_size=REGISTRY.TRAINMETA.batch_size)
        train_kwargs["train_loader"] = train_loader

    model = get_model()
    losses = model.train(**train_kwargs)

    save_path = REGISTRY.PATH.model / f"model-{model.id}.pkl"
    save_path = save_path.resolve()

    typer.echo(f"Saving model to {save_path}")
    model.save(save_path)
    REGISTRY.add_blob(save_path)
    REGISTRY.FLOWSYNTH.active_model_path = save_path
    REGISTRY.commit()

    db.add(losses)


@app.command()
def test_model(model_path: Optional[str] = typer.Option(None)):
    """
    Test the trained model.
    """
    from src.config.base import REGISTRY
    from src.database.dataset import FlowSynthDataset
    from src.database.factory import DBFactory
    from src.flow_synthesizer.api import evaluate_inference
    from src.flow_synthesizer.base import ModelWrapper

    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.active_model_path if model_path is None else model_path
    )

    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()
    test_dataset = FlowSynthDataset(db, test_flag=True)

    losses = evaluate_inference(model, test_dataset.audio_bridges)
    db.add(losses)


@app.command()
def estimate_synth_params(
    model_path: Optional[str] = typer.Option(None),
    audio_path: str = typer.Option(...),
    # patch_output: str = typer.Option(...),
):
    """
    Estimate synth parameters from an audio signal.
    """
    import torch

    from src.config.base import REGISTRY
    from src.database.dataset import load_formatted_audio
    from src.daw.factory import SynthHostFactory
    from src.flow_synthesizer.base import ModelWrapper

    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.active_model_path if model_path is None else model_path
    )
    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))
    synth_host = sh_factory()

    formatted_signal, signal = load_formatted_audio(  # TODO: Report confidence
        audio_path
    )
    with torch.no_grad():
        params = model(formatted_signal)

    synth_host.set_patch(params[0].tolist())
    print(synth_host.get_patch_as_model())  # TODO : save as .fxp


if __name__ == "__main__":
    app()
