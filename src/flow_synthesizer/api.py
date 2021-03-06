import json
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

from src.config.base import REGISTRY
from src.config.paths import get_project_root
from src.config.registry_sections import FlowSynthSection, TrainMetadataSection
from src.database.dataset import FlowSynthDataset, load_formatted_audio
from src.database.factory import DBFactory
from src.daw.audio_model import AudioBridgeTable
from src.daw.factory import SynthHostFactory
from src.daw.signal_processing import spectral_convergence, spectral_mse
from src.flow_synthesizer.base import ModelWrapper
from src.flow_synthesizer.factory import ModelFactory
from src.midi.generation import mono_midi
from src.utils.loss_model import LossTable, TrainValTestEnum


def prepare_registry(dataset: FlowSynthDataset, commit: bool = False) -> None:
    """
    Prepare the registry with default values for the model training and evaluation.
    """
    if REGISTRY.TRAINMETA is None:
        REGISTRY.TRAINMETA = TrainMetadataSection(
            in_dim=dataset.in_dim,
            out_dim=dataset.out_dim,
        )

    if REGISTRY.FLOWSYNTH is None:
        REGISTRY.FLOWSYNTH = FlowSynthSection()

    if commit:
        REGISTRY.commit()


def get_model(
    hyper_parameters: Optional[FlowSynthSection] = None,
    dataset: Optional[FlowSynthDataset] = None,
) -> ModelWrapper:

    if hyper_parameters is None:
        if REGISTRY.FLOWSYNTH is None:
            raise ValueError("REGISTRY.FLOWSYNTH section is not set.")
        hyper_parameters = REGISTRY.FLOWSYNTH

    if dataset is not None:
        in_dim = dataset.in_dim
        out_dim = dataset.out_dim
    else:
        in_dim = REGISTRY.TRAINMETA.in_dim
        out_dim = REGISTRY.TRAINMETA.out_dim

    model_factory = ModelFactory(
        in_dim=in_dim,
        out_dim=out_dim,
        encoding_dim=hyper_parameters.encoding_dim,
        latent_dim=hyper_parameters.latent_dim,
        channels=hyper_parameters.channels,
        hidden_dim=hyper_parameters.hidden_dim,
        ae_base=hyper_parameters.ae_base,
        ed_layer=hyper_parameters.ed_layer,
        model=hyper_parameters.model,
        flow_type=hyper_parameters.flow_type,
        flow_length=hyper_parameters.flow_length,
        n_layers=hyper_parameters.n_layers,
        kernel=hyper_parameters.kernel,
        dilation=hyper_parameters.dilation,
        regressor=hyper_parameters.regressor,
        regressor_flow_type=hyper_parameters.regressor_flow_type,
        regressor_hidden_dim=hyper_parameters.regressor_hidden_dim,
        regressor_layers=hyper_parameters.regressor_layers,
        reconstruction_loss=hyper_parameters.reconstruction_loss,
        disentangling_model=hyper_parameters.disentangling_model,
        disentangling_layers=hyper_parameters.disentangling_layers,
        semantic_dim=-hyper_parameters.semantic_dim,
    )
    model = model_factory()
    return model


def _write_audio(bridge: AudioBridgeTable, signal: np.ndarray, replace_pattern: str):
    file_path = bridge.audio_path.replace(".pt", replace_pattern)
    wavfile.write(
        file_path,
        REGISTRY.SYNTH.sample_rate,
        signal,
    )
    REGISTRY.add_blob(file_path)


def evaluate_inference(
    model: ModelWrapper,
    audio_bridges: list[AudioBridgeTable],
    write_audio: bool = False,
    monophonic: bool = True,
) -> list[LossTable]:
    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))
    synth_host = sh_factory()

    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()

    losses = []

    for bridge in tqdm(audio_bridges):
        formatted_signal, target_signal = load_formatted_audio(bridge.audio_path)
        with torch.no_grad():
            estimated_params = model(formatted_signal)[0]

        if write_audio:
            _write_audio(bridge, target_signal.cpu().numpy(), ".wav")

        if monophonic:
            synth_params = db.get_synth_params(bridge)
            midi_file_path = mono_midi(as_file_path=True)

            synth_host.set_patch(synth_params)
            target_signal = synth_host.render(midi_file_path)
            synth_host.set_patch(estimated_params.tolist())
            inferred_audio = synth_host.render(midi_file_path)

            if write_audio:
                _write_audio(bridge, target_signal, "_mono.wav")

        else:
            synth_host.set_patch(estimated_params.tolist())
            inferred_audio = synth_host.render(bridge.midi_path)

        if write_audio:
            _write_audio(bridge, inferred_audio, "_inferred.wav")

        for loss_callable in (spectral_convergence, spectral_mse):
            loss = loss_callable(inferred_audio, target_signal)

            loss_model = LossTable(
                model_id=str(model.id),
                type=str(loss_callable.__name__),
                train_val_test=TrainValTestEnum.TEST,
                value=loss,
            )

            losses.append(loss_model)

    return losses


def load_diva_presets() -> list[dict]:
    dataset_path = (
        get_project_root()
        / "src"
        / "flow_synthesizer"
        / "acids_ircam_flow_synthesizer"
        / "code"
        / "dataset.json"
    ).resolve()

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    return [preset["MIDI"] for preset in dataset.values()]
