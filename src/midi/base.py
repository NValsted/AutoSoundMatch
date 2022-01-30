from dataclasses import dataclass
from random import randint

from mido import Message, MidiFile, MidiTrack


@dataclass
class MidiNote:
    note: int
    velocity: int
    length: int
    offset: int

    @classmethod
    def random(
        cls,
        note_range: tuple[int, int] = (21, 108),
        velocity_range: tuple[int, int] = (0, 127),
        length_range : tuple[int, int] = (1, 256),
        offset_range : tuple[int, int] = (1, 256),
    ) -> "MidiNote":
        return cls(
            note=randint(*note_range),
            velocity=randint(*velocity_range),
            length=randint(*length_range),
            offset=randint(*offset_range),
        )

    def on_message(self):
        return Message("note_on", note=self.note, velocity=self.velocity, time=self.offset)

    def off_message(self):
        return Message(
            "note_off", note=self.note, velocity=self.velocity, time=self.length + self.offset
        )


def generate_midi(filepath: str):
    file = MidiFile(type=0)
    track = MidiTrack()
    file.tracks.append(track)

    for _ in range(100):
        note = MidiNote.random()
        track.append(note.on_message())
        track.append(note.off_message())

    file.save(filepath)


def partition_midi(polyphony_ratio=0.5, max_voices=8):
    pass
