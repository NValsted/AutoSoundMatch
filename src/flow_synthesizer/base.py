import inspect
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID, uuid4
from warnings import warn

import torch
from sqlalchemy.exc import IntegrityError
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.base import PYTORCH_DEVICE, REGISTRY
from src.database.dataset import DatasetParamsTable
from src.database.factory import DBFactory
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.loss import (
    multinomial_loss,
    multinomial_mse_loss,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.vae.ae import (
    DisentanglingAE,
    RegressionAE,
)
from src.flow_synthesizer.checkpoint import (
    FlowSynthParamsTable,
    ModelCheckpointTable,
    TrainMetadataParamsTable,
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
        self.model.to(PYTORCH_DEVICE)

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
            self.loss = nn.MSELoss(reduction="mean").to(PYTORCH_DEVICE)
        elif loss.value == "l1":
            self.loss = nn.SmoothL1Loss(reduction="mean").to(PYTORCH_DEVICE)
        elif loss.value == "bce":
            self.loss = nn.BCELoss(reduction="mean").to(PYTORCH_DEVICE)
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
        time_limit: Optional[int] = None,
        cp_callback: Optional[Callable[["ModelWrapper", list[LossTable]], None]] = None,
        *args,
        **kwargs,
    ) -> list[LossTable]:
        losses = []

        datetime_start = datetime.utcnow()

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
                    device=PYTORCH_DEVICE,
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

                if self._accumulated_epochs >= REGISTRY.TRAINMETA.start_regress or (
                    not isinstance(self.model, RegressionAE)
                    and not isinstance(self.model, DisentanglingAE)
                ):
                    self.scheduler.step(validation_loss.value)

            if cp_callback is not None:
                cp_callback(self, losses)

            self._accumulated_epochs += 1

            if time_limit is not None:
                if (datetime.utcnow() - datetime_start) > timedelta(minutes=time_limit):
                    warn(
                        f"Time limit reached at {datetime.utcnow()} for model"
                        f" {self.id}. Stopping training."
                    )
                    break

        return losses

    def evaluate(self, evaluation_loader: DataLoader) -> LossTable:
        eval_kwargs = dict(
            loader=evaluation_loader,
            args=AttributeWrapper(
                device=PYTORCH_DEVICE,
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

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "ae_model"):
            with torch.no_grad():
                _, embedding, _ = self.model.ae_model(x)
            return embedding
        else:
            raise ValueError("Model does not encode to a latent space.")

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return self.model(x)

    def _make_checkpoint(
        self, losses: list[LossTable], path: Optional[Path] = None
    ) -> Path:
        validation_losses = [
            loss
            for loss in losses
            if loss.train_val_test == TrainValTestEnum.VALIDATION
        ]
        if len(validation_losses) > 0:
            latest_val_loss = validation_losses[-1].value
        else:
            latest_val_loss = None

        if path is None:
            cp_path = (
                REGISTRY.PATH.model
                / f"model-{self.id}_cp-{self._accumulated_epochs}.pkl"
            ).resolve()
        else:
            cp_path = path.resolve()

        flow_synth_params = FlowSynthParamsTable.from_registry_section(
            REGISTRY.FLOWSYNTH
        )
        train_metadata_params = TrainMetadataParamsTable.from_registry_section(
            REGISTRY.TRAINMETA
        )
        dataset_params = DatasetParamsTable.from_registry_section(REGISTRY.DATASET)

        model_checkpoint_meta = ModelCheckpointTable(
            model_id=str(self.id),
            checkpoint_path=str(cp_path),
            accumulated_epochs=self._accumulated_epochs,
            val_loss=latest_val_loss,
            flow_synth_params=flow_synth_params.id,
            train_metadata_params=train_metadata_params.id,
            dataset_params=dataset_params.id,
        )

        with cp_path.open("wb") as f:
            pickle.dump(self, f)
        REGISTRY.add_blob(cp_path)

        db_factory = DBFactory(engine_url=REGISTRY.DATABASE.url)
        db = db_factory()

        for entry in [flow_synth_params, train_metadata_params, dataset_params]:
            try:
                db.safe_add([entry])
            except IntegrityError:
                pass  # warn(str(e))
        db.safe_add([model_checkpoint_meta])

        return cp_path

    @staticmethod
    def default_cp_callback(model: "ModelWrapper", losses: list[LossTable]) -> None:
        """
        Default checkpoint behaviour which saves model when validation loss is lowest.
        """
        validation_losses = [
            loss
            for loss in losses
            if loss.train_val_test == TrainValTestEnum.VALIDATION
        ]
        previous_losses, latest_loss = validation_losses[:-1], validation_losses[-1]

        if len(previous_losses) > 0:  # and model._accumulated_epochs % 10 == 0:
            min_val_loss = min(previous_losses, key=lambda x: x.value)
            print("MAKING CHECKPOINT", min_val_loss, latest_loss)
            if latest_loss.value < min_val_loss.value:
                model._make_checkpoint(losses)

    def save(self, path: Path) -> None:
        self._make_checkpoint([], path)

    @staticmethod
    def load(path: Path) -> "ModelWrapper":
        with path.open("rb") as f:
            model = pickle.load(f)

        if not isinstance(model, ModelWrapper):
            raise ValueError(
                f"Object at path {path} was expected to be a ModelWrapper object, but"
                f" got {type(model)}"
            )

        return model
