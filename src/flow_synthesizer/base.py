from enum import Enum
from dataclasses import dataclass

from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.ae import AE, DisentanglingAE, RegressionAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.vae import VAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.wae import WAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.vae_flow import VAEFlow
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.basic import GatedCNN, GatedMLP


class ModelEnum(Enum):
    GatedMLP = GatedMLP
    GatedCNN = GatedCNN
    AE = AE
    RegressionAE = RegressionAE
    DisentanglingAE = DisentanglingAE
    VAE = VAE
    WAE = WAE
    VAEFlow = VAEFlow


class FlowTypeEnum(str, Enum):
    planar = "planar"
    sylvester = "sylvester"
    real_nvp = "real_nvp"
    maf = "maf"
    iaf = "iaf"
    dsf = "dsf"
    ddsf = "ddsf"
    ddsf_iaf = "ddsf_iaf"
    iaf_ctx = "iaf_ctx"
    maf_ctx = "maf_ctx"


@dataclass
class ModelWrapper:
    model: ModelEnum
