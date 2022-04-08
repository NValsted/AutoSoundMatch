from src.config.paths import get_project_root
from src.config.registry_sections import SynthSection

synth_config = SynthSection(
    synth_path=get_project_root() / "data" / "synth" / "TAL-NoiseMaker.vst3",
    locked_parameters={
        24: 0.0,  # Osc 1 Waveform duplicate
        37: 0.5,  # Osc 2 Phase duplicate
        39: 0.5,  # Osc 2 Phase duplicate
        "-": 0.5,
        "Master Volume": 0.40800002217292786,
        "Fitler Keyfollow": 0.0,
        "Osc 2 Volume": 0.0,
        "Osc Mastertune": 0.5,
        "Osc 2 Tune": 0.5,
        "Osc 2 Fine Tune": 0.5,
        "Osc Sync": 0.0,
        "Lfo 2 Waveform": 0.0,
        "Lfo 2 Rate": 0.0,
        "Lfo 2 Amount": 0.5,
        "Lfo 2 Destination": 0.0,
        "Lfo 1 Phase": 0.0,
        "Lfo 2 Phase": 0.0,
        "Osc 2 FM": 0.0,
        "Osc 2 Phase": 0.5,
        "Osc 2 PW": 0.5,
        "Transpose": 0.5,
        "Free Ad Attack": 0.0,
        "Free Ad Decay": 0.0,
        "Free Ad Amount": 0.0,
        "Free Ad Destination": 0.0,
        "Lfo 1 Sync": 0.0,
        "Lfo 2 Sync": 0.0,
        "Lfo 2 Keytrigger": 0.0,
        "Portamento Amount": 0.0,
        "Portamento Mode": 0.0,
        "Voices": 1.0,
        "Velocity Volume": 0.0,
        "Velocity Contour": 0.0,
        "Velocity Filter": 0.0,
        "Pitchwheel Cutoff": 0.0,
        "Pitchwheel Pitch": 0.0,
        "Ringmodulation": 0.0,
        "Chorus 2 Enable": 0.0,
        "Reverb Pre Delay": 0.0,
        "Reverb High Cut": 0.0,
        "Reverb Low Cut": 1.0,
        "Master Detune": 0.0,
        "Panic": 0.0,
        "MIDI LEARN": 0.0,
        "Envelope Destination": 0.0,
        "Envelope Speed": 0.0,
        "Envelope Amount": 0.0,
        "Envelope One Shot Mode": 0.0,
        "Envelope Fix Tempo": 0.0,
        "Envelope Reset": 0.0,
        "Delay Sync": 0.0,
        "Delay x2 L": 0.0,
        "Delay x2 R": 0.0,
        "Delay High Shelf": 0.0,
        "Delay Low Shelf": 0.0,
        "MIDI Clear": 0.0,
        "MIDI Lock": 0.0,
        "Bypass": 0.0,
    },
)
