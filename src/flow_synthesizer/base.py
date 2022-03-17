import inspect
from dataclasses import dataclass, field
from os import PathLike
from typing import BinaryIO, Callable, Optional, Union
from uuid import UUID, uuid4

from torch import load, nn, save
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.base import REGISTRY
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.loss import (
    multinomial_loss,
    multinomial_mse_loss,
)
from src.flow_synthesizer.enums import LossEnum, ModelEnum, SchedulerModeEnum
from src.utils.loss_model import LossTable, TrainValTestEnum
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
    id: UUID = field(default_factory=uuid4)

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

    def _determine_loss_kwarg(self, func: Callable) -> str:
        for kwarg_name in ("loss_params", "loss"):
            if kwarg_name in inspect.signature(func).parameters:
                return kwarg_name
        else:
            raise ValueError(f"Unable to determine loss keyword argument in {func}")

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        validation_loader: Optional[DataLoader] = None,
        *args,
        **kwargs,
    ) -> list[LossTable]:
        # TODO: Use scheduler
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
            loss_kwarg = self._determine_loss_kwarg(self.model.train_epoch)
            train_kwargs[loss_kwarg] = self.loss

            loss = self.model.train_epoch(**train_kwargs)
            loss_model = LossTable(
                model_id=str(self.id),
                type=str(self.loss),
                train_val_test=TrainValTestEnum.TRAIN,
                value=loss.item(),
            )
            losses.append(loss_model)

            if validation_loader is not None:
                validation_loss = self.evaluate(validation_loader)
                losses.append(validation_loss)

            self._accumulated_epochs += 1

        return losses

    def evaluate(self, evaluation_loader: DataLoader) -> LossTable:
        eval_kwargs = dict(
            loader=evaluation_loader,
            args=AttributeWrapper(
                device="cpu",
            ),
        )
        loss_kwarg = self._determine_loss_kwarg(self.model.eval_epoch)
        eval_kwargs[loss_kwarg] = self.loss

        loss = self.model.eval_epoch(**eval_kwargs)
        loss_model = LossTable(
            model_id=str(self.id),
            type=str(self.loss),
            train_val_test=TrainValTestEnum.VALIDATION,
            value=loss.item(),
        )

        return loss_model

    def save(self, path: Union[str, PathLike, BinaryIO]) -> None:
        save(self.model, path)

    @classmethod
    def load(cls, path: Union[str, PathLike, BinaryIO]) -> None:
        _loaded_model = load(path)
        return cls(model=_loaded_model)
