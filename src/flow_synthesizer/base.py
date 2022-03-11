import inspect
from dataclasses import dataclass, field

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config.base import REGISTRY
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.loss import (
    multinomial_loss,
    multinomial_mse_loss,
)
from src.flow_synthesizer.enums import LossEnum, ModelEnum, SchedulerModeEnum
from src.utils.meta import AttributeWrapper


@dataclass
class ModelWrapper:
    model: ModelEnum
    trainable: bool = True
    _accumulated_epochs: int = 0
    optimizer: Adam = field(init=False)
    scheduler: ReduceLROnPlateau = field(init=False)
    loss: LossEnum = field(init=False)
    beta: float = field(init=False)
    gamma: float = field(init=False)
    delta: float = field(init=False)

    def __post_init__(self):
        if self.trainable:
            if REGISTRY.TRAINMETA is None:
                raise ValueError(
                    "REGISTRY.TRAINMETA is required to initialize model, but is not set"
                )
            self.prepare(
                lr=REGISTRY.TRAINMETA.learning_rate,
                scheduler_mode=REGISTRY.TRAINMETA.scheduler_mode,
                scheduler_factor=REGISTRY.TRAINMETA.scheduler_factor,
                scheduler_patience=REGISTRY.TRAINMETA.scheduler_patience,
                scheduler_verbose=REGISTRY.TRAINMETA.scheduler_verbose,
                scheduler_threshold=REGISTRY.TRAINMETA.scheduler_threshold,
                loss=REGISTRY.TRAINMETA.loss,
            )
            self._update_warmup()

    def prepare(
        self,
        lr: int,
        scheduler_mode: SchedulerModeEnum,
        scheduler_factor: float,
        scheduler_patience: int,
        scheduler_verbose: bool,
        scheduler_threshold: float,
        loss: LossEnum,
    ):
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=lr,
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=scheduler_verbose,
            threshold=scheduler_threshold,
        )

        if loss.value == "mse":
            self.loss = nn.MSELoss(reduction="mean")  # TODO: device
        elif loss.value == "l1":
            self.loss = nn.SmoothL1Loss(reduction="mean")  # TODO: device
        elif loss.value == "bce":
            self.loss = nn.BCELoss(reduction="mean")  # TODO: device
        elif loss.value == "multinomial":
            self.loss = multinomial_loss
        elif loss.value == "multi_mse":
            self.loss = multinomial_mse_loss
        else:
            raise ValueError(f"Loss {loss} is invalid.")

    def _update_warmup(
        self,
        beta_factor: float = REGISTRY.TRAINMETA.beta_factor
        if REGISTRY.TRAINMETA is not None
        else 1.0,
        reg_factor: float = REGISTRY.TRAINMETA.reg_factor
        if REGISTRY.TRAINMETA is not None
        else 1e3,
        start_regress: int = REGISTRY.TRAINMETA.start_regress
        if REGISTRY.TRAINMETA is not None
        else 1e2,
        warm_regress: int = REGISTRY.TRAINMETA.warm_regress
        if REGISTRY.TRAINMETA is not None
        else 1e2,
        warm_latent: int = REGISTRY.TRAINMETA.warm_latent
        if REGISTRY.TRAINMETA is not None
        else 50,
        epoch: int = 0,
        *args,
        **kwargs,
    ):
        self.beta = beta_factor * (float(epoch) / float(max(warm_latent, epoch)))
        self.gamma = 0
        if epoch >= start_regress:
            self.gamma = (float(epoch - start_regress) * reg_factor) / float(
                max(warm_regress, epoch - start_regress)
            )
        self.delta = 0

    def train(
        self,
        train_loader,
        epochs: int = REGISTRY.TRAINMETA.epochs
        if REGISTRY.TRAINMETA is not None
        else 1,
        *args,
        **kwargs,
    ) -> list[float]:

        losses = []
        for _ in tqdm(range(epochs)):
            self._update_warmup(
                *args,
                **kwargs,
                epoch=self._accumulated_epochs,
            )

            train_kwargs = dict(
                loader=train_loader,
                optimizer=self.optimizer,
                args=AttributeWrapper(
                    device="cpu",
                    beta=self.beta,
                    gamma=self.gamma,
                ),
            )

            for kwarg_name in ("loss_params", "loss"):
                if kwarg_name in inspect.signature(self.model.train_epoch).parameters:
                    train_kwargs[kwarg_name] = self.loss
                    break
            else:
                raise ValueError(
                    "Unable to determine loss keyword argument in"
                    f" {self.model.train_epoch}"
                )

            loss = self.model.train_epoch(**train_kwargs)
            losses.append(loss)

            self._accumulated_epochs += 1

        return losses
