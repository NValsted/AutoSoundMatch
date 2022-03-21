import inspect
import json
import random
from glob import glob
from typing import Optional

import torch
import typer
from librosa.util import valid_audio
from librosa.util.exceptions import ParameterError
from sqlmodel import select
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config.base import PYTORCH_DEVICE, REGISTRY, Registry
from src.config.registry_sections import RegistrySectionsEnum, RegistrySectionsMap
from src.database.dataset import FlowSynthDataset, load_formatted_audio
from src.database.factory import DBFactory
from src.daw.audio_model import AudioBridge, AudioBridgeTable  # NOQA: F401
from src.daw.factory import SynthHostFactory
from src.daw.render_model import RenderParams, RenderParamsTable  # NOQA: F401
from src.daw.synth_model import SynthParams, SynthParamsTable  # NOQA: F401
from src.flow_synthesizer.api import evaluate_inference, get_model, prepare_registry
from src.flow_synthesizer.base import ModelWrapper
from src.midi.generation import generate_midi
from src.utils.signal_processing import process_sample

app = typer.Typer()


@app.command()
def register() -> None:
    """
    Interactive interface to manually register values in the registry.
    """
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
    for k, v in dict(REGISTRY).items():
        typer.echo(f"{k}: {v}")
    # TODO : display info about blobs - number of blobs, combined size of blobs, etc.


@app.command()
def reset():
    """
    Reset the registry to default values, drop tables, and remove generated data.
    """
    global REGISTRY

    if REGISTRY.DATABASE is not None:
        db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
        db = db_factory()
        db.drop_tables()

    REGISTRY.clear_blobs()
    REGISTRY = Registry()
    REGISTRY.commit()


@app.command()
def setup_relational_models(
    synth_path: str = typer.Option(...), engine_url: Optional[str] = typer.Option(None)
) -> None:
    """
    Create tables in a local database for storing audio files and VST
    parameters.
    """

    db_factory_kwargs = dict()
    if engine_url is not None:
        db_factory_kwargs["engine_url"] = engine_url

    render_params = RenderParams()
    sh_factory_kwargs = dict(synth_path=synth_path)
    sh_factory_kwargs.update(
        {
            k: v
            for k, v in dict(render_params).items()
            if k in inspect.signature(SynthHostFactory).parameters
        }
    )

    db_factory = DBFactory(**db_factory_kwargs)
    sh_factory = SynthHostFactory(**sh_factory_kwargs)

    synth_host = sh_factory()
    definition_path = synth_host.create_parameter_table()
    typer.echo(
        f"Created model and table definition for {synth_path} at {definition_path}"
    )

    db = db_factory()
    db.create_tables()

    db_factory.register(commit=True)
    sh_factory.register(commit=True)


@app.command()
def generate_param_tuples(
    midi_path: str = typer.Option(...),
    audio_path: str = typer.Option(...),
    patches_per_midi: int = typer.Option(5),
) -> None:
    """
    Generate tuples of audio files with corresponding midi files, render
    parameters and parameters from a VST instrument.
    """

    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))
    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)

    synth_host = sh_factory()
    db = db_factory()
    render_params = RenderParams()

    generate_midi(midi_path)
    for midi_file_path in tqdm(glob(f"{midi_path}/.generated/*.mid")):
        midi_file_path = midi_file_path.replace("\\", "/")
        for i in range(patches_per_midi):
            synth_host.set_random_patch()
            audio = synth_host.render(midi_file_path, render_params)
            audio_file_path = midi_file_path.replace(".mid", f"_{i}.pt").replace(
                midi_path, audio_path
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
                audio_path=audio_file_path,
                midi_path=midi_file_path,
                synth_params=synth_params.id,
                test_flag=True if random.random() < 0.1 else False,
            )
            db.add([synth_params])
            db.add([audio_bridge])


@app.command()
def process_audio():
    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()

    with db.session() as session:
        query = select(AudioBridgeTable).where()
        audio_bridges = session.exec(query).all()

    for bridge in tqdm(audio_bridges):
        signal = torch.load(bridge.audio_path).to(PYTORCH_DEVICE)
        processed = process_sample(signal)

        save_path = bridge.audio_path.replace(".pt", "_processed.pt")
        bridge.processed_path = save_path

        torch.save(processed, save_path)
        REGISTRY.add_blob(save_path)

    db.add(audio_bridges)


@app.command()
def train_model(
    model_dir: str = typer.Option(...),
    validation_split: Optional[float] = typer.Option(0.15),
) -> None:
    """
    Train a model to estimate parameters.
    """
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

    save_path = f"{model_dir}/model-{model.id}.pkl"
    typer.echo(f"Saving model to {save_path}")
    model.save(save_path)
    REGISTRY.add_blob(save_path)
    REGISTRY.FLOWSYNTH.latest_model_path = save_path
    REGISTRY.commit()

    db.add(losses)


@app.command()
def test_model(model_path: Optional[str] = typer.Option(None)):
    """
    Test the trained model.
    """
    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.latest_model_path if model_path is None else model_path
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
    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.latest_model_path if model_path is None else model_path
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
