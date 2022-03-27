import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from shutil import rmtree
from typing import Optional, Union

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from scipy.stats import randint, truncnorm
from tqdm import tqdm

from src.config.base import REGISTRY
from src.midi.base import ASMMidiNote, ASMMidiTrack
from src.utils.distributions.base import EmpiricalDistribution


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

        save_path = f"{result_dir}/{i}.mid"
        file.save(save_path)
        REGISTRY.add_blob(save_path)


@dataclass(frozen=True)
class MidiSnippetProperties:
    polyphony_ratio: Optional[float] = 0.5
    max_voices: Optional[int] = 8


@dataclass
class MidiSnippetDistributions:
    polyphony: EmpiricalDistribution
    voices: EmpiricalDistribution


class MidiPartition:
    class BoundaryStrategy(str, Enum):
        EXCLUDE = "EXCLUDE"
        TRIM = "TRIM"

    strategy: BoundaryStrategy
    _total_ticks: int
    _messages: list[Union[Message, MetaMessage]]
    _active_notes: dict = defaultdict(deque)
    _tick_acc: int = 0

    def __init__(
        self,
        total_ticks: int,
        strategy: BoundaryStrategy,
        meta_messages: Optional[list[MetaMessage]] = None,
    ):
        self._messages = []
        self.strategy = strategy
        self._total_ticks = total_ticks
        if meta_messages is not None:
            self._messages.extend(meta_messages)

    def _trim(self):
        remaining_ticks = self._total_ticks - self._tick_acc
        for on_messages in self._active_notes.values():
            while on_messages:
                on_message = on_messages.popleft()
                off_message = Message(
                    type="note_off",
                    time=int(remaining_ticks),
                    note=on_message.note,
                    velocity=on_message.velocity,
                )
                remaining_ticks = 0
                self._messages.append(off_message)

    def parse_message(self, message_queue: deque[Message]):
        message = message_queue.popleft()

        if self._tick_acc + message.time > self._total_ticks:
            message.time = int(message.time - self._total_ticks + self._tick_acc)
            message_queue.appendleft(message)

            if self.strategy == MidiPartition.BoundaryStrategy.TRIM:
                self._trim()

            raise StopIteration

        self._tick_acc += message.time

        if message.type == "note_on":
            self._active_notes[message.note].append(message)

        elif message.type == "note_off":
            if len(self._active_notes[message.note]) > 0:
                self._active_notes[message.note].popleft()

        self._messages.append(message)

    def finalize(self) -> Optional[MidiFile]:
        for message in self._messages:
            if message.type == "note_on":
                break
        else:
            return

        track = MidiTrack()
        track.extend(self._messages)
        file = MidiFile()
        file.tracks.append(track)
        return file

    @classmethod
    def partition_file(
        cls, file: MidiFile, strategy: BoundaryStrategy
    ) -> list["MidiPartition"]:
        partitions = []

        duration = REGISTRY.SYNTH.duration
        tempo = bpm2tempo(REGISTRY.SYNTH.bpm)
        ticks_per_partition = second2tick(duration, file.ticks_per_beat, tempo)

        active_partition = MidiPartition(
            total_ticks=ticks_per_partition,
            strategy=strategy,
            meta_messages=[MetaMessage("set_tempo", tempo=tempo)],
        )

        for track in file.tracks:
            message_queue = deque(track)
            while len(message_queue) > 0:
                try:
                    active_partition.parse_message(message_queue)
                except StopIteration:
                    partitions.append(active_partition)
                    active_partition = MidiPartition(
                        total_ticks=ticks_per_partition,
                        strategy=strategy,
                        meta_messages=[MetaMessage("set_tempo", tempo=tempo)],
                    )

        if strategy == MidiPartition.BoundaryStrategy.TRIM:
            active_partition._trim()
        partitions.append(active_partition)

        return partitions


def partition_midi(
    midi_files: list[MidiFile],
    properties: MidiSnippetProperties = MidiSnippetProperties(),
) -> list[MidiFile]:
    """
    This takes a collection of MIDI files and partitions them into
    fixed-size snippets with desired properties.
    """

    partitioned_files = []

    for file in tqdm(midi_files):
        partitions = [
            partition.finalize()
            for partition in MidiPartition.partition_file(
                file, strategy=MidiPartition.BoundaryStrategy.TRIM
            )
        ]
        partitions = [partition for partition in partitions if partition is not None]
        # TODO : filter partitions based on MidiSnippetProperties

        partitioned_files.extend(partitions)

    return partitioned_files
