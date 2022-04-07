from src.config.paths import get_project_root
from src.config.registry_sections import SynthSection

synth_config = SynthSection(
    synth_path=get_project_root() / "data" / "synth" / "MikaMicro64.dll",
    locked_parameters={
        "Voice mode": 0.0,
        "Glide speed": 0.0,
        "Oscillator 2 waveform": 0.0,
        "Oscillator 2 coarse": 0.0,
        "Oscillator 2 fine": 0.0,
        "Oscillator 2 split": 0.0,
    },
)
