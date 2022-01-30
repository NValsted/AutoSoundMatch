from dataclasses import dataclass

from dawdreamer import RenderEngine

from src.daw.base import SynthHost
from src.utils.temporary_context import temporary_attrs


@dataclass
class SynthHostFactory:
    synth_path: str
    sample_rate: int = 44100 
    buffer_size: int = 128
    bpm: int = 128

    def __call__(self, *args, **kwargs) -> SynthHost:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            engine = RenderEngine(tmp.sample_rate, tmp.buffer_size)
            engine.set_bpm(tmp.bpm)
            synth = engine.make_plugin_processor("synth", tmp.synth_path)
            return SynthHost(engine=engine, synth=synth, **kwargs)
