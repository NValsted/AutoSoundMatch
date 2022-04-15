from src.config.paths import get_project_root
from src.config.registry_sections import SynthSection, TrainMetadataSection
from src.flow_synthesizer.enums import LossEnum, SchedulerModeEnum

train_metadata_config = TrainMetadataSection(
    in_dim=[128, 87],
    out_dim=32,
    epochs=500,
    batch_size=64,
    time_limit=None,
    loss=LossEnum.mse,
    learning_rate=2e-4,
    scheduler_mode=SchedulerModeEnum.min,
    scheduler_factor=0.5,
    scheduler_patience=20,
    scheduler_verbose=True,
    scheduler_threshold=1e-7,
    beta_factor=1.0,
    reg_factor=1e3,
    start_regress=1e2,
    warm_regress=1e2,
    warm_latent=50,
)

synth_config = SynthSection(
    synth_path=get_project_root() / "data" / "synth" / "MikaMicro64.dll",
    locked_parameters={
        "Voice mode": 0.0,
        "Glide speed": 0.0,
        "Oscillator 2 waveform": 0.0,
        "Oscillator 2 coarse": 0.0,
        "Oscillator 2 fine": 0.0,
        "Oscillator 2 split": 0.0,
    },
)
