from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class AudioBridge(SQLModel):
    audio_path: str = Field(primary_key=True)
    midi_path: str = Field(primary_key=True)
    render_params: str = Field(foreign_key="RenderParams.id", index=True, nullable=True)
    synth_params: str = Field(foreign_key="SynthParams.id", index=True, nullable=True)
    test_flag: bool = Field(default=False)


class AudioBridgeTable(AudioBridge, table=True):
    __tablename__ = "AudioBridge"
    __table_args__ = (
        UniqueConstraint("audio_path", "midi_path", "render_params", "synth_params"),
    )
