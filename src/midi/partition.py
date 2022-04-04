from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from tqdm import tqdm

from src.config.base import REGISTRY


@dataclass(frozen=True)
class MidiProperties:
    voices_distribution: Optional[dict] = None
    max_silence_ratio: Optional[float] = None  # per partition ratio of silence
    min_voices: Optional[int] = None
    max_voices: Optional[int] = None


@dataclass
class AnalyzedPartition:
    partition: MidiFile
    voice_distribution: dict


class UnconformingPartitionError(Exception):
    pass


class MidiPartition:
    class BoundaryStrategy(str, Enum):
        EXCLUDE = "EXCLUDE"
        TRIM = "TRIM"

    strategy: BoundaryStrategy
    _active_notes: dict
    _total_ticks: int
    _messages: list[Union[Message, MetaMessage]]
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
        self._active_notes = defaultdict(deque)
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

    def parse_message(self, message_deque: deque[Message]):
        message = message_deque.popleft()

        if self._tick_acc + message.time > self._total_ticks:
            message.time = int(message.time - self._total_ticks + self._tick_acc)
            message_deque.appendleft(message)

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
            return None

        track = MidiTrack()
        track.extend(self._messages)
        file = MidiFile()
        file.tracks.append(track)
        return file

    @property
    def tick_acc(self):
        return self._tick_acc

    @property
    def active_notes(self):
        return self._active_notes

    @classmethod
    def partition_file(
        cls, file: MidiFile, strategy: BoundaryStrategy
    ) -> list["MidiPartition"]:
        partitions = []

        duration = REGISTRY.SYNTH.duration - 1
        tempo = bpm2tempo(REGISTRY.SYNTH.bpm)
        ticks_per_partition = second2tick(duration, file.ticks_per_beat, tempo)

        active_partition = MidiPartition(
            total_ticks=ticks_per_partition,
            strategy=strategy,
            meta_messages=[MetaMessage("set_tempo", tempo=tempo)],
        )

        for track in file.tracks:
            message_deque = deque(track)
            while len(message_deque) > 0:
                try:
                    active_partition.parse_message(message_deque)
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

    @classmethod
    def analyze_partition(cls, file: MidiFile, properties: MidiProperties) -> dict:
        voices_distribution = defaultdict(int)
        prev_tick_acc = 0

        partition_processor = cls(
            total_ticks=float("inf"),
            strategy=MidiPartition.BoundaryStrategy.TRIM,
        )
        message_deque = deque(file.tracks[0])

        while len(message_deque) > 0:
            try:
                partition_processor.parse_message(message_deque)
                if partition_processor.tick_acc != prev_tick_acc:
                    ticks = partition_processor.tick_acc - prev_tick_acc
                    num_voices = len(partition_processor.active_notes)

                    if (
                        properties.min_voices is not None
                        and num_voices < properties.min_voices
                    ):
                        raise UnconformingPartitionError

                    if (
                        properties.max_voices is not None
                        and num_voices > properties.max_voices
                    ):
                        raise UnconformingPartitionError

                    voices_distribution[num_voices] += ticks
                    prev_tick_acc = partition_processor.tick_acc

            except StopIteration:
                if len(message_deque) != 0:
                    raise RuntimeError(
                        f"{len(message_deque)} messages left in message deque"
                    )

        if properties.max_silence_ratio is not None and (
            voices_distribution[0]
            > properties.max_silence_ratio * partition_processor.tick_acc
        ):
            raise UnconformingPartitionError

        return voices_distribution


def partition_midi(
    midi_files: list[MidiFile],
    properties: MidiProperties = MidiProperties(),
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
        partitioned_files.extend(partitions)

    analyzed_partitions = []
    for file in partitioned_files:
        try:
            result = MidiPartition.analyze_partition(file, properties)
            analyzed_partitions.append(
                AnalyzedPartition(partition=file, voice_distribution=result)
            )
        except UnconformingPartitionError:
            continue

    if properties.voices_distribution is not None:
        raise NotImplementedError
        # total_distribution = defaultdict(int)
        # for analyzed_partition in analyzed_partitions:
        #     for num_voices, ticks in analyzed_partition.voice_distribution.items():
        #         total_distribution[num_voices] += ticks
        # total_ticks = sum(total_distribution.values())
        # normalized_distribution = {
        #     k: v / total_ticks for k, v in total_distribution.items()
        # }
        # diff, key = max(
        #     (normalized_distribution.get(k, 0) - v, k)
        #     for k, v in properties.voices_distribution.items()
        # )

    return [ap.partition for ap in analyzed_partitions]
