from typing import Optional
from glob import glob
import json

import typer

from src.config.base import Registry
from src.config.registry_sections import RegistrySectionsEnum, RegistrySectionsMap
from src.database.factory import DBFactory
from src.daw.factory import SynthHostFactory
from src.midi.generation import generate_midi
# from src.utils.flp_wrapper import Project

app = typer.Typer()


@app.command()
def register() -> None:
    """
    Interactive interface to manually register values in the registry.
    """
    registry = Registry()
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
            raise ValueError(f"Invalid entry: {entry}")

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
        registry.__setattr__(k, sectionModel)
    
    registry.commit()


@app.command()
def inspect_registry():
    """
    Display the current registry values.
    """
    raise NotImplementedError


@app.command()
def reset():
    """
    Reset the registry to default values and remove generated data.
    """
    raise NotImplementedError


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
    
    db_factory = DBFactory(**db_factory_kwargs)
    sh_factory = SynthHostFactory(synth_path=synth_path)
    
    synth_host = sh_factory()
    definition_path = synth_host.create_parameter_table()
    typer.echo(f"Created model and table definition for {synth_path} at {definition_path}")

    from src.daw.render_model import RenderParamsTable
    from src.daw.synth_model import SynthParamsTable
    from src.daw.audio_model import AudioBridgeTable
    db = db_factory()
    db.create_tables()
    db_factory.register()


@app.command()
def generate_param_triples(
    midi_path: str = typer.Option(...),
    patches_per_midi: int = typer.Option(100),
) -> None:
    """
    Generate triples of audio files with corresponding midi files and
    parameters from a VST instrument.
    """

    sh_factory = SynthHostFactory()
    db_factory = DBFactory()

    synth_host = sh_factory()
    db = db_factory()

    generate_midi(midi_path)
    for file in glob(f"{midi_path}/*.mid"):
        for _ in range(patches_per_midi):
            pass


@app.command()
def extract_midi_from_flp() -> None:
    """
    Extract MIDI files from FLP files.
    """
    raise NotImplementedError
    # Project("data/flp/simple.flp")


if __name__ == "__main__":
    app()
