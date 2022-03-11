from dataclasses import InitVar, dataclass, field
from enum import Enum
from io import BufferedReader, BytesIO, FileIO
from struct import unpack


class ChunkType(str, Enum):
    header = "FLhd"
    events = "FLdt"


class PatternEvent(int, Enum):
    pass


class MIDIEvent(int, Enum):
    FLP_NOTE_ON = 1
    FLP_MIDI_CHAN = 4
    FLP_MIDI_NOTE = 5
    FLP_MIDI_PATCH = 6
    FLP_MIDI_BANK = 7
    FLP_MIDI_CTRLS = (3 << 6) + 16
    FLP_REMOTE_CTRL_MIDI = (3 << 6) + 34


def get_event_size(stream):
    if (b := unpack("B", stream.read(1))[0]) is not None:
        data_len = b & 0x7F
        shift = 7
        while (b & 0x80) != 0:
            b = unpack("B", stream.read(1))[0]
            if b is None:
                return None
            data_len |= (b & 0x7F) << shift
            shift += 7
        return data_len
    return None


@dataclass
class Event:
    type: MIDIEvent
    content: bytes

    def __init__(self, stream: BufferedReader):
        self.type = (int.from_bytes(stream.read(1), byteorder="little") & 0xF0) >> 4

        size_map = [
            lambda _: 1,
            lambda _: 2,
            lambda _: 4,
            get_event_size,
        ]
        size = size_map[self.type >> 6](stream)
        self.content = stream.read(size)


@dataclass
class Header:
    format: int
    channel_count: int
    beat_division: int

    @classmethod
    def from_bytes(cls, content: bytes) -> "Header":
        header = unpack("<HHH", content)
        return cls(format=header[0], channel_count=header[1], beat_division=header[2])


@dataclass
class Events:
    events: list[Event]

    @classmethod
    def from_bytes(cls, content: bytes) -> "Events":
        events = list()

        with BufferedReader(BytesIO(content)) as stream:
            stream_size = stream.seek(0, 2)
            stream.seek(0, 0)
            while stream.tell() < stream_size:
                event = Event(stream)
                events.append(event)

        return cls(events=events)


@dataclass
class Project:
    """
    Minimal python object wrapper for FLP file format.
    """

    filepath: InitVar[str]
    header: Header = field(init=False)
    events: Events = field(init=False)

    def __post_init__(self, filepath: str) -> None:
        with BufferedReader(FileIO(filepath, "rb")) as stream:
            stream.seek(0, 0)
            self.header = Header.from_bytes(self._read_chunk(stream, ChunkType.header))
            self.events = Events.from_bytes(self._read_chunk(stream, ChunkType.events))

    @staticmethod
    def _read_chunk(stream: BufferedReader, chunk_type: ChunkType) -> bytes:
        decoded_chunk_type = stream.read(4).decode("utf-8")
        assert decoded_chunk_type == chunk_type.value
        chunk_size = unpack("<I", stream.read(4))[0]
        print("CHUNK SIZE", chunk_size)
        return stream.read(chunk_size)
