import random
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Union
from uuid import uuid4

from mido import MidiFile, bpm2tempo, second2tick
from scipy.stats import randint, truncnorm
from tqdm import tqdm

from src.config.base import REGISTRY
from src.midi.base import ASMMidiNote, ASMMidiTrack


def generate_midi(
    target_dir: Optional[Union[str, Path]] = None, number_of_files: int = 50
):
    if target_dir is not None:
        result_dir = Path(target_dir).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = REGISTRY.PATH.midi

    scales = ["major", "minor", "blues pentatonic minor", "chromatic", "harmonic major"]
    octave_distributions = [
        truncnorm(-1, 1, loc=0, scale=0.75),
        truncnorm(-2, 2, loc=0, scale=2),
        randint(low=-2, high=2),
    ]
    length_distributions = [
        truncnorm(6, 256, loc=128, scale=32),
        truncnorm(8, 512, loc=256, scale=16),
        randint(low=32, high=512),
    ]
    offset_distributions = [
        truncnorm(0, 2048, loc=1024, scale=512),
        randint(low=0, high=2048),
    ]
    number_of_notes = truncnorm(2, 16, loc=4, scale=8)
    quantization_intervals = [32, 64, 128, 256]

    for i in tqdm(range(number_of_files)):
        notes = [
            ASMMidiNote.random_from_scale(
                scale=random.choice(scales),
                octave_distribution=random.choice(octave_distributions),
                length_distribution=random.choice(length_distributions),
                offset_distribution=random.choice(offset_distributions),
            )
            for _ in range(int(number_of_notes.rvs()))
        ]
        quantized_notes = ASMMidiTrack.quantize(
            notes, interval=random.choice(quantization_intervals)
        )
        track = ASMMidiTrack.from_notes(quantized_notes)

        file = MidiFile(type=0)
        file.tracks.append(track)

        save_path = result_dir / f"generated_{i}.mid"
        save_path = save_path.resolve()
        file.save(save_path)
        REGISTRY.add_blob(save_path)


def mono_midi(note: int = 60, as_file_path: bool = False) -> MidiFile:
    """
    Generate a MIDI partition with a single MIDI note.
    """
    file = MidiFile(type=0)

    duration = REGISTRY.SYNTH.duration - 1
    tempo = bpm2tempo(REGISTRY.SYNTH.bpm)
    ticks_per_partition = second2tick(duration, file.ticks_per_beat, tempo)

    midi_note = ASMMidiNote(
        note=note, velocity=100, length=ticks_per_partition, offset=0
    )
    track = ASMMidiTrack.from_notes([midi_note])
    file.tracks.append(track)

    if as_file_path:
        mono_file_path = Path(gettempdir()) / f"{uuid4()}.mid"
        file.save(str(mono_file_path))
        return mono_file_path

    return file
