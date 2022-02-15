import os
import random
from shutil import rmtree

from tqdm import tqdm
from mido import MidiFile
from scipy.stats import truncnorm, randint

from src.midi.base import ASMMidiNote, ASMMidiTrack


def generate_midi(target_dir: str, number_of_files: int = 50):
    result_dir = f"{target_dir}/.generated"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if os.path.exists(result_dir):
        rmtree(result_dir)
    os.mkdir(result_dir)
    
    scales = ["major", "minor", "blues pentatonic minor", "chromatic", "harmonic major"]
    octave_distributions = [
        truncnorm(-1, 1, loc=0, scale=0.75),
        truncnorm(-2, 2, loc=0, scale=2),
        randint(low=-2, high=2),
    ]
    length_distributions = [
        truncnorm(6, 256, loc=128, scale=32),
        truncnorm(8, 512, loc=256, scale=16),
        randint(low=32, high=512)
    ]
    offset_distributions = [
        truncnorm(0, 2048, loc=1024, scale=512),
        randint(low=0, high=2048)
    ]
    number_of_notes = truncnorm(4, 64, loc=8, scale=8)
    quantization_intervals = [32, 64, 128, 256]

    for i in tqdm(range(number_of_files)):
        notes = [
            ASMMidiNote.random_from_scale(
                scale=random.choice(scales),
                octave_distribution=random.choice(octave_distributions),
                length_distribution=random.choice(length_distributions),
                offset_distribution=random.choice(offset_distributions),
            ) for _ in range(int(number_of_notes.rvs()))
        ]
        quantized_notes = ASMMidiTrack.quantize(
            notes, interval=random.choice(quantization_intervals)
        )
        track = ASMMidiTrack.from_notes(quantized_notes)
        
        file = MidiFile(type=0)
        file.tracks.append(track)
        file.save(f"{result_dir}/{i}.mid")


def partition_midi(polyphony_ratio=0.5, max_voices=8):
    """
    This takes a collection of MIDI files and partitions them into
    fixed-size snippets with desired properties.
    """
    raise NotImplementedError