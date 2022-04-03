from typing import Optional

import typer

app = typer.Typer()


@app.command()
def setup_paths(
    downloads: Optional[str] = typer.Option(None),
    presets: Optional[str] = typer.Option(None),
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
    if presets is not None:
        path_kwargs["presets"] = presets
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
def update_registry(
    fixture: str = typer.Argument(
        ...,
        help=(
            "Path to fixture file (Optionally append ::{class_name} at end of"
            " file name to load a specific class)"
        ),
    )
):
    import re
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    from src.config.base import REGISTRY
    from src.config.registry_sections import RegistrySectionsMap

    class AmbiguousConfigError(ValueError):
        pass

    node_match = re.search(r"::\s*[A-Za-z_]\w*\s*$", fixture)
    if node_match:
        node = node_match.group(0).strip()[2:]
        path = Path(fixture[: node_match.start()]).resolve()
    else:
        node = None
        path = Path(fixture).resolve()

    spec = spec_from_file_location(path.name, path.as_posix())
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if node is not None:
        if not hasattr(module, node):
            raise ValueError(f"No config class named {node} found in {path}")

        section_config = getattr(module, node)
        section_name = REGISTRY.classify_section(section_config)
        setattr(REGISTRY, section_name, section_config)

    else:
        section_matches = {k: [] for k in RegistrySectionsMap.keys()}
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            section_config = getattr(module, attr)

            try:
                section_name = REGISTRY.classify_section(section_config)
            except ValueError:
                continue

            section_matches[section_name].append(section_config)
            if len(section_matches[section_name]) > 1:
                raise AmbiguousConfigError(
                    f"Multiple {section_name} sections found in {path} - Resolve"
                    " ambiguity by separating into multiple files or using"
                    " ::{class_name} at end of file to specify a specific class"
                )

        for section_name, section_config in section_matches.items():
            if len(section_config) == 1:
                setattr(REGISTRY, section_name, section_config[0])

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
    from warnings import warn

    from mido import MidiFile
    from tqdm import tqdm

    from src.config.base import REGISTRY
    from src.midi.partition import MidiProperties, partition_midi

    for dir in directory:
        file_paths = Path(dir).rglob("*.mid")

        midi_files = []
        for i, path in enumerate(file_paths):
            try:
                midi_files.append(MidiFile(path))
            except OSError as e:
                warn(f"OSError: Could not open {path}: {e}")
            except ValueError as e:
                warn(f"ValueError: Could not open {path}: {e}")
            except Exception as e:
                warn(f"Could not open {path}: {e}")
            if i > 500:
                break

        if len(midi_files) == 0:
            warn(f"No midi files found in directory {dir}")
            continue

        partitioned_files = partition_midi(
            midi_files, properties=MidiProperties(max_silence_ratio=0.4, max_voices=8)
        )

        for file in tqdm(partitioned_files):
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
    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))

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
def generate_param_triples(
    num_presets: Optional[int] = typer.Option(500),
    num_midi: Optional[int] = typer.Option(500),
    pairs: Optional[int] = typer.Option(10),
):
    """
    Generate triples of audio files with corresponding midi files and
    parameters from a VST instrument.
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

    midi_paths = list(REGISTRY.PATH.midi.glob("*.mid"))

    if num_midi is None:
        num_midi = len(midi_paths)

    if pairs is None:
        pairs = num_midi

    if len(midi_paths) < num_midi:
        typer.echo(f"Generating {num_midi - len(midi_paths)} additional midi files")
        generate_midi(midi_paths, number_of_files=num_midi - len(midi_paths))
        midi_paths = list(REGISTRY.PATH.midi.glob("*.mid"))

    preset_paths = list(REGISTRY.PATH.presets.glob("*.fxp"))
    presets = [synth_host.load_preset(path) for path in preset_paths]

    if num_presets is None:
        num_presets = len(presets)

    if len(presets) < num_presets:
        typer.echo(f"Generating {num_presets - len(presets)} additional presets")
        presets.extend(
            [
                [param[-1] for param in synth_host.random_patch()]
                for _ in range(num_presets - len(presets))
            ]
        )

    random.shuffle(midi_paths)
    random.shuffle(presets)
    midi_paths = midi_paths[:num_midi]
    presets = presets[:num_presets]

    for i, preset in enumerate(tqdm(presets, leave=True)):

        synth_host.set_patch(preset)
        synth_params = synth_host.get_patch_as_model(table=True)
        synth_params_id = synth_params.id
        db.safe_add([synth_params])

        for j in tqdm(range(pairs), leave=False):
            midi_file_path = midi_paths[(i + j) % len(midi_paths)]

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

            audio_bridge = AudioBridgeTable(
                audio_path=str(audio_file_path),
                midi_path=str(midi_file_path),
                synth_params=synth_params_id,
                test_flag=True if random.random() < 0.1 else False,
            )

            db.safe_add([audio_bridge])


@app.command()
def process_audio(
    chunk_size: int = typer.Option(512),
    reprocess: bool = typer.Option(False),
    num_workers: int = typer.Option(4),
):
    from multiprocessing.pool import ThreadPool
    from pathlib import Path

    import torch
    from sqlalchemy import func
    from sqlmodel import select
    from tqdm import tqdm

    from src.config.base import PYTORCH_DEVICE, REGISTRY
    from src.database.factory import DBFactory
    from src.daw.audio_model import AudioBridgeTable
    from src.daw.synth_model import SynthParamsTable  # NOQA: F401
    from src.utils.signal_processing import SignalProcessor

    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()

    if reprocess:
        query = select(AudioBridgeTable)
    else:
        query = select(AudioBridgeTable).where(
            AudioBridgeTable.processed_path.is_(None)
        )

    finalized_instances = []

    def _save(args) -> AudioBridgeTable:
        processed, bridge = args
        save_path = REGISTRY.PATH.processed_audio / Path(bridge.audio_path).name
        bridge.processed_path = str(save_path.resolve())

        torch.save(processed, save_path)
        REGISTRY.add_blob(save_path)
        return bridge

    with db.session() as session:
        total_audio_bridges = session.exec(
            select(func.count()).select_from(query.subquery())
        ).all()[0]

    for offset in tqdm(range(0, total_audio_bridges, chunk_size), leave=True):
        with db.session() as session:
            audio_bridges = session.exec(query.limit(chunk_size).offset(offset)).all()

        signals = [
            torch.load(bridge.audio_path).to(PYTORCH_DEVICE) for bridge in audio_bridges
        ]

        if PYTORCH_DEVICE.type == "cpu":
            processed = SignalProcessor.concurrent_batch_process(
                signals, num_workers=num_workers
            )
        else:
            processed = SignalProcessor.batch_process(signals)

        with ThreadPool() as p:
            finalized_instances.extend(p.map(_save, zip(processed, audio_bridges)))

    db.safe_add(finalized_instances)


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
    typer.echo(synth_host.get_patch_as_model())  # TODO : save as .fxp


if __name__ == "__main__":
    app()
