from typing import Optional

import typer

app = typer.Typer()


def model_selection(model_id: Optional[str] = None, latest: bool = False):
    from sqlmodel import select

    from src.config.base import REGISTRY
    from src.database.factory import DBFactory
    from src.flow_synthesizer.base import ModelWrapper
    from src.utils.loss_model import LossTable

    if model_id is None:
        if latest:
            active_model_path = REGISTRY.FLOWSYNTH.active_model_path
            model = ModelWrapper.load(active_model_path)
            model_id = str(model.id)

        else:
            db = DBFactory(engine_url=REGISTRY.DATABASE.url)()
            with db.session() as session:
                model_ids = list(session.exec(select(LossTable.model_id).distinct()))
                formatted = "\n".join(
                    [f"{i}: {m_id}" for i, m_id in enumerate(model_ids)]
                )
                response = typer.prompt(
                    f"Available models (provide number)\n{formatted}"
                )
                if response.isdigit() and int(response) < len(model_ids):
                    model_id = model_ids[int(response)]
                else:
                    raise ValueError(f"Invalid response: {response}")

    elif latest:
        typer.echo(
            "Latest flag is provided as well as explicit model id - using model-id"
        )

    return model_id


@app.command()
def train_val_loss(
    model_id: Optional[str] = typer.Option(None), latest: bool = typer.Option(False)
):
    from src.graphics.aggregate_loss import train_val_loss

    model_id = model_selection(model_id, latest)

    fig = train_val_loss(model_id=model_id)
    fig.show()


@app.command()
def spectral_loss_distplot(
    model_id: Optional[str] = typer.Option(None), latest: bool = typer.Option(False)
):
    from src.graphics.aggregate_loss import spectral_loss_distplot

    model_id = model_selection(model_id, latest)

    fig = spectral_loss_distplot(model_id=model_id)
    fig.show()


@app.command()
def pca_latent_space(model_path: Optional[str] = typer.Option(None)):
    from src.config.base import REGISTRY
    from src.flow_synthesizer.base import ModelWrapper
    from src.graphics.latent_space import pca_latent_space

    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.active_model_path if model_path is None else model_path
    )
    fig = pca_latent_space(model)
    fig.show()


@app.command()
def tsne_latent_space(model_path: Optional[str] = typer.Option(None)):
    from src.config.base import REGISTRY
    from src.flow_synthesizer.base import ModelWrapper
    from src.graphics.latent_space import tsne_latent_space

    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.active_model_path if model_path is None else model_path
    )
    fig = tsne_latent_space(model)
    fig.show()


@app.command()
def inference_comparison(model_path: Optional[str] = typer.Option(None)):
    from src.config.base import REGISTRY
    from src.flow_synthesizer.base import ModelWrapper
    from src.graphics.inference_instance import inference_comparison

    model = ModelWrapper.load(
        REGISTRY.FLOWSYNTH.active_model_path if model_path is None else model_path
    )
    fig = inference_comparison(model)
    fig.show()


if __name__ == "__main__":
    app()
