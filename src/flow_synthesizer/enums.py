from enum import Enum
from functools import partial

from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.basic import (
    GatedCNN,
    GatedMLP,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.ae import (
    AE,
    DisentanglingAE,
    RegressionAE,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.vae import VAE
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.vae_flow import (
    VAEFlow,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.wae import WAE


class AEBaseModelEnum(Enum):
    AE = AE
    VAE = VAE
    WAE = WAE
    VAEFlow = VAEFlow


class EDLayerEnum(str, Enum):
    mlp = "mlp"
    gated_mlp = "gated_mlp"
    cnn = "cnn"
    res_cnn = "res_cnn"
    gated_cnn = "gated_cnn"


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


class ModelEnum(str, Enum):
    MLP = "MLP"
    GatedMLP = "GatedMLP"
    CNN = "CNN"
    ResCNN = "ResCNN"
    GatedCNN = "GatedCNN"
    RegressionAE = "RegressionAE"
    DisentanglingAE = "DisentanglingAE"


ModelEnumConstructorMap = dict(
    MLP=partial(GatedMLP, type_mod="normal"),
    GatedMLP=partial(GatedMLP, type_mod="gated"),
    CNN=partial(GatedCNN, type_mod="normal"),
    ResCNN=partial(GatedCNN, type_mod="residual"),
    GatedCNN=partial(GatedCNN, type_mod="gated"),
    RegressionAE=RegressionAE,
    DisentanglingAE=DisentanglingAE,
)


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
    bce = "bce"
    multinomial = "multinomial"
    multi_mse = "multi_mse"


class DisentanglingModelEnum(str, Enum):
    density = "density"
    base = "base"
    full = "full"


class SchedulerModeEnum(str, Enum):
    min = "min"
    max = "max"
