# THIS FILE WAS AUTOMATICALLY GENERATED AT 2022-03-17T17:55:23.446244
#####################################################################

from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint

from src.utils.meta import hash_field_to_uuid


class SynthParams(SQLModel):
    id: Optional[str] = Field(primary_key=True, default=None)
    Oscillator_1_waveform: float = Field(alias="0")
    Oscillator_1_coarse: float = Field(alias="1")
    Oscillator_1_fine: float = Field(alias="2")
    Oscillator_1_split: float = Field(alias="3")
    Oscillator_2_waveform: float = Field(alias="4")
    Oscillator_2_coarse: float = Field(alias="5")
    Oscillator_2_fine: float = Field(alias="6")
    Oscillator_2_split: float = Field(alias="7")
    Oscillator_mix: float = Field(alias="8")
    FM_mode: float = Field(alias="9")
    FM_coarse: float = Field(alias="10")
    FM_fine: float = Field(alias="11")
    Filter_enabled: float = Field(alias="12")
    Filter_cutoff: float = Field(alias="13")
    Filter_resonance: float = Field(alias="14")
    Filter_key_tracking: float = Field(alias="15")
    Volume_envelope_attack: float = Field(alias="16")
    Volume_envelope_decay: float = Field(alias="17")
    Volume_envelope_sustain: float = Field(alias="18")
    Volume_envelope_release: float = Field(alias="19")
    Volume_envelope_velocity_sensiti: float = Field(alias="20")
    Modulation_envelope_attack: float = Field(alias="21")
    Modulation_envelope_decay: float = Field(alias="22")
    Modulation_envelope_sustain: float = Field(alias="23")
    Modulation_envelope_release: float = Field(alias="24")
    Modulation_envelope_velocity_sen: float = Field(alias="25")
    Vibrato_amount: float = Field(alias="26")
    Vibrato_frequency: float = Field(alias="27")
    Vibrato_delay: float = Field(alias="28")
    Volume_envelope_to_FM_amount: float = Field(alias="29")
    Volume_envelope_to_filter_cutoffhz: float = Field(alias="30")
    Modulation_envelope_to_FM_amountsemitones: float = Field(alias="31")
    Modulation_envelope_to_filter_cuhz: float = Field(alias="32")
    Vibrato_to_FM_amount: float = Field(alias="33")
    Vibrato_to_filter_cutoff: float = Field(alias="34")
    Voice_mode: float = Field(alias="35")
    Glide_speed: float = Field(alias="36")
    Master_volume: float = Field(alias="37")

    class Config:
        validate_all = True

    _auto_uuid = hash_field_to_uuid("id")


class SynthParamsTable(SynthParams, table=True):
    __tablename__ = "SynthParams"
    __table_args__ = (UniqueConstraint(
        "Oscillator_1_waveform",
        "Oscillator_1_coarse",
        "Oscillator_1_fine",
        "Oscillator_1_split",
        "Oscillator_2_waveform",
        "Oscillator_2_coarse",
        "Oscillator_2_fine",
        "Oscillator_2_split",
        "Oscillator_mix",
        "FM_mode",
        "FM_coarse",
        "FM_fine",
        "Filter_enabled",
        "Filter_cutoff",
        "Filter_resonance",
        "Filter_key_tracking",
        "Volume_envelope_attack",
        "Volume_envelope_decay",
        "Volume_envelope_sustain",
        "Volume_envelope_release",
        "Volume_envelope_velocity_sensiti",
        "Modulation_envelope_attack",
        "Modulation_envelope_decay",
        "Modulation_envelope_sustain",
        "Modulation_envelope_release",
        "Modulation_envelope_velocity_sen",
        "Vibrato_amount",
        "Vibrato_frequency",
        "Vibrato_delay",
        "Volume_envelope_to_FM_amount",
        "Volume_envelope_to_filter_cutoffhz",
        "Modulation_envelope_to_FM_amountsemitones",
        "Modulation_envelope_to_filter_cuhz",
        "Vibrato_to_FM_amount",
        "Vibrato_to_filter_cutoff",
        "Voice_mode",
        "Glide_speed",
        "Master_volume"
    ),)
