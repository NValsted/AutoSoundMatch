from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class AudioBridge(SQLModel):
    audio_path: str = Field(primary_key=True)
    midi_path: str = Field(primary_key=True)
    render_params: int = Field(foreign_key="RenderParams.id", index=False, nullable=True)
    synth_params: int = Field(foreign_key="SynthParams.id", index=False, nullable=True)


class AudioBridgeTable(AudioBridge, table=True):
    __tablename__ = "AudioBridge"
    __table_args__ = (UniqueConstraint("render_params", "synth_params"),)
