from dataclasses import dataclass

from dawdreamer import RenderEngine

from src.daw.base import SynthHost
from src.utils.temporary_context import temporary_attrs
from src.config.base import REGISTRY
from src.config.registry_sections import SynthSection


@dataclass
class SynthHostFactory:
    synth_path: str
    sample_rate: int
    buffer_size: int
    duration: float
    bpm: int

    def __call__(self, *args, **kwargs) -> SynthHost:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            engine = RenderEngine(tmp.sample_rate, tmp.buffer_size)
            engine.set_bpm(tmp.bpm)
            synth = engine.make_plugin_processor("synth", tmp.synth_path)
            engine.load_graph([(synth, [])])
            return SynthHost(engine=engine, synth=synth, **kwargs)

    def register(self, commit: bool = False) -> None:
        REGISTRY.SYNTH = SynthSection(
            synth_path=self.synth_path,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            bpm=self.bpm,
        )
        if commit:
            REGISTRY.commit()
