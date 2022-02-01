from copy import deepcopy
from typing import Union
import random
import json

from scipy.stats import (
    _continuous_distns as continuous_distributions,
    _discrete_distns as discrete_distributions,
    rv_continuous,
    rv_discrete
)
from mido import Message, MidiTrack
from pydantic import BaseModel, validator

with open("src/midi/scales.json", "r") as f:
    SCALES = json.load(f)


class ASMMidiNote(BaseModel):
    """
    Convenience class for generating MIDI note messages.
    """
    note: int
    velocity: int
    length: int
    offset: int

    @validator("note", allow_reuse=True)
    @validator("velocity", allow_reuse=True)
    def valid_midi_value(cls, value):
        if value < 0 or value > 127:
            raise ValueError(f"MIDI value must be between 0 and 127 (inclusive). Got {value=}")
        return value

    @validator("length", allow_reuse=True)
    @validator("offset", allow_reuse=True)
    def valid_time(cls, value):
        if value < 0:
            raise ValueError(f"Time must be greater than or equal to 0. Got {value=}")
        return value

    @classmethod
    def random_from_scale(
        cls,
        scale: Union[list[int], str] = "chromatic",
        base_note: int = 60,
        octave_distribution: Union[rv_continuous, rv_discrete]
            = continuous_distributions.truncnorm(-2, 2, loc=0, scale=2),
        velocity_distribution: Union[rv_continuous, rv_discrete]
            = continuous_distributions.truncnorm(0, 127, loc=64, scale=5),
        length_distribution: Union[rv_continuous, rv_discrete]
            = discrete_distributions.randint(low=0, high=5),
        offset_distribution: Union[rv_continuous, rv_discrete]
            = discrete_distributions.randint(low=0, high=100),
    ) -> "ASMMidiNote":
        """
        Instantiate a random midi note from a scale.
        """
        if isinstance(scale, str):
            if scale not in SCALES:
                raise ValueError(
                    f"{scale} is not a valid scale. Valid scales are {', '.join(SCALES.keys())}"
                )
            scale = SCALES[scale]

        note = base_note + (random.choice(scale) - 1) + int(octave_distribution.rvs()) * 12
        velocity = int(velocity_distribution.rvs())
        length = int(length_distribution.rvs())
        offset = int(offset_distribution.rvs())
        
        return cls(note=note, velocity=velocity, length=length, offset=offset)

    def on_message(self):
        """
        Returns note_on message from timestamp 0.
        """
        return Message("note_on", note=self.note, velocity=self.velocity, time=self.offset)

    def off_message(self):
        """
        Returns note_off message from timestamp 0.
        """
        return Message(
            "note_off", note=self.note, velocity=self.velocity, time=self.length + self.offset
        )


class ASMMidiTrack(MidiTrack):
    """
    Convenience class for generating MIDI tracks.
    """
    @classmethod
    def from_notes(cls, notes: list[ASMMidiNote]) -> "ASMMidiTrack":
        track = cls()
        messages = [
            message for note in notes for message in (note.on_message(), note.off_message())
        ]
        messages.sort(key=lambda x: x.time)

        timestamp = 0
        for message in messages:
            message.time = message.time - timestamp
            timestamp += message.time
            track.append(message)
        return track

    @staticmethod
    def quantize(notes: list[ASMMidiNote], interval: int) -> list[ASMMidiNote]:
        notes = deepcopy(notes)
        for note in notes:
            note.length = round(note.length / interval) * interval
            note.offset = round(note.offset / interval) * interval
        return notes
