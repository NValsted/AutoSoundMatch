from enum import Enum
from dataclasses import dataclass

from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.ae import AE, DisentanglingAE, RegressionAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.vae import VAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.wae import WAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.vae_flow import VAEFlow
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.basic import GatedCNN, GatedMLP


class AEBaseModelEnum(Enum):
    AE = AE
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


class ModelEnum(Enum):
    GatedMLP = GatedMLP
    GatedCNN = GatedCNN
    RegressionAE = RegressionAE
    DisentanglingAE = DisentanglingAE


class RegressorEnum(str, Enum):
    mlp = "mlp"
    bnn = "bnn"
    flow = "flow"
    flow_p = "flow_p"
    flow_trans = "flow_trans"
    flow_m = "flow_m"
    flow_kl = "flow_kl"
    flow_kl_f = "flow_kl_f"
    flow_cde = "flow_cde"
    flow_ext = "flow_ext"
    flow_post = "flow_post"
    flow_dec = "flow_dec"


class LossEnum(str, Enum):
    mse = "mse"
    l1 = "l1"
    multinomial = "multinomial"
    multi_mse = "multi_mse"


class DisentanglingModelEnum(str, Enum):
    density = "density"
    base = "base"
    full = "full"


@dataclass
class ModelWrapper:
    model: ModelEnum
