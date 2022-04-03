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
from src.daw.audio_model import AudioBridgeTable
from src.daw.factory import SynthHostFactory
from src.daw.synth_model import SynthParamsTable
from src.flow_synthesizer.base import ModelWrapper
from src.utils.signal_processing import SignalProcessor, StereoToMono


def inference_comparison(
    model: ModelWrapper, audio_bridge: Optional[AudioBridgeTable] = None
) -> Figure:

    if audio_bridge is None:
        db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
        db = db_factory()

        with db.session() as session:
            audio_bridge = session.exec(select(AudioBridgeTable).limit(1)).first()
            synth_params_query = select(SynthParamsTable.__table__).filter(
                SynthParamsTable.id.in_([audio_bridge.synth_params])
            )
            synth_params = session.execute(synth_params_query).first()

    sh_factory = SynthHostFactory(**dict(REGISTRY.SYNTH))

    formatted_signal, target_signal = load_formatted_audio(audio_bridge.audio_path)
    with torch.no_grad():
        estimated_params = model(formatted_signal)[0]

    synth_host = sh_factory()
    synth_host.set_patch(estimated_params.tolist())
    inferred_audio = torch.from_numpy(synth_host.render(audio_bridge.midi_path))
    processed_inferred_audio = SignalProcessor()(inferred_audio)

    param_x_axis = np.arange(0, estimated_params.shape[0])

    signal_x_axis = np.linspace(0, REGISTRY.SYNTH.duration, inferred_audio.shape[0])
    stereo_to_mono = StereoToMono((None, 2))

    fig = make_subplots(rows=3, cols=2, shared_yaxes=True)

    for i, color in enumerate(("rgb(123,17,58)", "rgba(123,17,58,100)")):
        fig.add_trace(
            go.Scatter(
                x=param_x_axis,
                y=synth_params[1:],
                mode="markers",
                marker_size=5 - (i * 2),
                marker_color=color,
            ),
            row=1,
            col=i + 1,
        )
    fig.add_trace(
        go.Scatter(
            x=param_x_axis,
            y=estimated_params,
            mode="markers",
            marker_size=5,
            marker_color="rgb(21,14,86)",
        ),
        row=1,
        col=2,
    )

    for param_idx in param_x_axis:
        fig.add_vline(
            x=param_idx,
            row=1,
            col="all",
            layer="below",
            line_color="rgba(100,100,100,200)",
            line_width=0.5,
        )

    fig.add_trace(
        go.Heatmap(z=formatted_signal.reshape(*(formatted_signal.shape[1:])).numpy()),
        row=2,
        col=1,
    )
    fig.add_trace(go.Heatmap(z=processed_inferred_audio.numpy()), row=2, col=2)

    fig.add_trace(
        go.Scatter(
            x=signal_x_axis,
            y=stereo_to_mono(target_signal).numpy(),
            line=dict(color="rgb(123,17,58)", width=1),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_x_axis,
            y=stereo_to_mono(inferred_audio).numpy(),
            line=dict(color="rgb(21,14,86)", width=1),
        ),
        row=3,
        col=2,
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_traces(showlegend=False)
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")

    fig["layout"]["yaxis1"].update(domain=[0.9, 1])
    fig["layout"]["yaxis2"].update(domain=[0.9, 1])
    fig["layout"]["yaxis3"].update(domain=[0.4, 0.85])
    fig["layout"]["yaxis4"].update(domain=[0.4, 0.85])
    fig["layout"]["yaxis5"].update(domain=[0, 0.35])
    fig["layout"]["yaxis6"].update(domain=[0, 0.35])

    return fig
