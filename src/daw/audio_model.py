from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class AudioBridge(SQLModel):
    audio_path: str = Field(primary_key=True)
    midi_path: str = Field(primary_key=True)
    synth_params: str = Field(foreign_key="SynthParams.id", index=True, nullable=True)
    test_flag: bool = Field(default=False)


class AudioBridgeTable(AudioBridge, table=True):
    __tablename__ = "AudioBridge"
    __table_args__ = (UniqueConstraint("audio_path", "midi_path", "synth_params"),)
