import numpy as np
import plotly.express as px
from plotly.graph_objs._figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sqlmodel import select
from tqdm import tqdm

from src.config.base import REGISTRY
from src.database.dataset import load_formatted_audio
from src.database.factory import DBFactory
from src.daw.audio_model import AudioBridgeTable
from src.flow_synthesizer.base import ModelWrapper


def _get_embeddings(model: ModelWrapper, max_datapoints: int) -> np.ndarray:
    db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
    db = db_factory()

    with db.session() as session:
        audio_bridges = list(
            session.exec(select(AudioBridgeTable).limit(max_datapoints))
        )

    processed_signals = [
        load_formatted_audio(bridge.audio_path)[0] for bridge in tqdm(audio_bridges)
    ]

    embeddings = np.array(
        [model.embed(entry)[0].cpu().numpy() for entry in tqdm(processed_signals)]
    )
    return embeddings


def pca_latent_space(model: ModelWrapper, max_datapoints: int = 10_000) -> Figure:
    """
    Computes and visualizes the 2-dimensional PCA reprensetation of the model's latent
    space
    """
    embeddings = _get_embeddings(model, max_datapoints)

    re_embeddings = PCA(n_components=2).fit_transform(embeddings)

    fig = px.scatter(x=re_embeddings[:, 0], y=re_embeddings[:, 1], title="PCA")
    return fig


def tsne_latent_space(
    model: ModelWrapper, max_datapoints: int = 10_000, pca: bool = True
) -> Figure:
    """
    Computes and visualizes a 2-dimensional t-SNE reprensetation of the model's latent
    space
    """
    embeddings = _get_embeddings(model, max_datapoints)

    re_embeddings = TSNE(
        n_components=2, init="pca" if pca else "random", learning_rate="auto"
    ).fit_transform(embeddings)

    fig = px.scatter(x=re_embeddings[:, 0], y=re_embeddings[:, 1], title="t-SNE")
    return fig
