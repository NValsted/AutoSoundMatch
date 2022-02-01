from typing import Optional

import typer

from src.database.factory import DBFactory
from src.daw.factory import SynthHostFactory
from src.midi.generation import generate_midi
# from src.utils.flp_wrapper import Project

app = typer.Typer()


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


@app.command()
def generate_param_triples(
    midi_path: str = typer.Option(...)
) -> None:
    """
    Generate triples of audio files with corresponding midi files and
    parameters from a VST instrument.
    """
    generate_midi(midi_path)


@app.command()
def extract_midi_from_flp() -> None:
    """
    Extract MIDI files from FLP files.
    """
    raise NotImplementedError()
    # Project("data/flp/simple.flp")


if __name__ == "__main__":
    app()
