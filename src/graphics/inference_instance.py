from typing import Optional

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
from sqlmodel import select

from src.config.base import REGISTRY
from src.database.dataset import load_formatted_audio
from src.database.factory import DBFactory
from src.daw.audio_model import AudioBridge, AudioBridgeTable
from src.daw.factory import SynthHostFactory
from src.flow_synthesizer.base import ModelWrapper
from src.utils.signal_processing import SignalProcessor, StereoToMono


def inference_comparison(
    model: ModelWrapper, audio_bridge: Optional[AudioBridge] = None
) -> Figure:

    if audio_bridge is None:
        db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
        db = db_factory()

        with db.session() as session:
            audio_bridge = session.exec(select(AudioBridgeTable).limit(1)).first()

    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))

    formatted_signal, target_signal = load_formatted_audio(audio_bridge.audio_path)
    with torch.no_grad():
        estimated_params = model(formatted_signal)[0]

    synth_host = sh_factory()
    synth_host.set_patch(estimated_params.tolist())
    inferred_audio = torch.from_numpy(synth_host.render(audio_bridge.midi_path))
    processed_inferred_audio = SignalProcessor()(inferred_audio)

    x_axis = np.linspace(0, REGISTRY.SYNTH.duration, inferred_audio.shape[0])
    stereo_to_mono = StereoToMono((None, 2))

    fig = make_subplots(rows=3, cols=2, shared_yaxes=True)
    fig.add_trace(
        go.Heatmap(z=formatted_signal.reshape(*(formatted_signal.shape[1:])).numpy()),
        row=2,
        col=1,
    )
    fig.add_trace(go.Heatmap(z=processed_inferred_audio.numpy()), row=2, col=2)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=stereo_to_mono(target_signal).numpy(),
            line=dict(color="firebrick", width=1),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=stereo_to_mono(inferred_audio).numpy(),
            line=dict(color="firebrick", width=1),
        ),
        row=3,
        col=2,
    )

    fig.update_traces(showlegend=False)

    return fig
