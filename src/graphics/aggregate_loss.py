import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots

from src.config.base import REGISTRY
from src.utils.loss_model import LossTable, TrainValTestEnum


def train_val_loss(model_id: str) -> Figure:
    df = pd.read_sql_table(LossTable.__tablename__, REGISTRY.DATABASE.url)

    model_loss = df[df["model_id"] == model_id]
    model_loss = model_loss.sort_values(by="time")

    train_loss = model_loss[
        model_loss["train_val_test"] == TrainValTestEnum.TRAIN.value
    ]
    validation_loss = model_loss[
        model_loss["train_val_test"] == TrainValTestEnum.VALIDATION.value
    ]

    train_loss_type = train_loss["type"].unique()
    validation_loss_type = validation_loss["type"].unique()

    if train_loss_type.size > 1:
        raise ValueError(f"Multiple loss types found in train loss: {train_loss_type}")
    elif validation_loss_type.size > 1:
        raise ValueError(
            f"Multiple loss types found in validation loss: {validation_loss_type}"
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=train_loss["time"],
            y=train_loss["value"],
            name="Train loss",
            line_color="rgb(123,17,58)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=validation_loss["time"],
            y=validation_loss["value"],
            name="Validation loss",
            line_color="rgb(21,151,187)",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=f"Loss evolution for model {model_id}",
        plot_bgcolor="rgb(246,246,246)",
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(
        title_text=f"<b>Train</b> loss - {train_loss_type[0]}", secondary_y=False
    )
    fig.update_yaxes(
        title_text=f"<b>Validation</b> loss - {validation_loss_type[0]}",
        secondary_y=True,
    )

    return fig


def spectral_loss_distplot(model_id: str) -> Figure:
    df = pd.read_sql_table(LossTable.__tablename__, REGISTRY.DATABASE.url)

    model_loss = df[df["model_id"] == model_id]
    test_loss = model_loss[model_loss["train_val_test"] == TrainValTestEnum.TEST.value]

    SC = test_loss[test_loss["type"] == "spectral_convergence"]
    MSE = test_loss[test_loss["type"] == "spectral_mse"]

    sc_fig = ff.create_distplot([SC["value"]], ["Spectral convergence"], bin_size=0.1)
    mse_fig = ff.create_distplot([MSE["value"]], ["Spectral MSE"], bin_size=1)

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=("Spectral convergence", "Spectral MSE"),
        horizontal_spacing=0.05,
    )

    for i, (sub_fig, data, rgb) in enumerate(
        ((sc_fig, SC, (123, 17, 58)), (mse_fig, MSE, (21, 14, 86)))
    ):
        primary_color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        secondary_color = f"rgb({rgb[0]-15},{rgb[1]-10},{rgb[2]-30})"

        fig.add_trace(
            go.Histogram(sub_fig["data"][0], marker_color=primary_color),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Scatter(
                sub_fig["data"][1],
                line_color=secondary_color,
            ),
            row=1,
            col=i + 1,
        )
        data["rug"] = 1.1
        fig.add_trace(
            go.Scatter(
                x=data["value"],
                y=data["rug"],
                mode="markers",
                marker=dict(color=primary_color, symbol="line-ns-open"),
            ),
            row=2,
            col=i + 1,
        )
        fig.update_yaxes(
            range=[1, 1.2],
            tickfont=dict(color="rgba(0,0,0,0)", size=14),
            row=2,
            col=i + 1,
        )

        for val, text, position in (
            (data["value"].mean(), "Mean", "top right"),
            (data["value"].median(), "Median", "top left"),
        ):
            fig.add_vline(
                x=val,
                col=i + 1,
                row=1,
                annotation_text=text,
                annotation_position=position,
                line_dash="dot",
                line_color=secondary_color,
            )

    fig.update_layout(showlegend=False, plot_bgcolor="rgb(246,246,246)")
    fig["layout"]["yaxis1"].update(domain=[0.1, 1])
    fig["layout"]["yaxis3"].update(domain=[0, 0.05])
    fig["layout"]["yaxis2"].update(domain=[0.1, 1])
    fig["layout"]["yaxis4"].update(domain=[0, 0.05])

    return fig
