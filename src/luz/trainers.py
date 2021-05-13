from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Union

import luz
import torch

__all__ = ["Trainer"]


Device = Union[str, torch.device]
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Trainer:
    def __init__(
        self,
        loss: Optional[Loss] = None,
        optimizer: Optional[luz.Optimizer] = None,
        start_epoch: Optional[int] = 1,
        stop_epoch: Optional[int] = 2,
        early_stopping: Optional[bool] = False,
        handlers: Optional[Iterable[luz.Handler]] = None,
        patience: Optional[int] = 5,
        log_filepath: Optional[str] = None,
        **loader_kwargs: Union[int, bool, luz.Transform],
    ) -> None:
        """Algorithm to train a model using data.

        Parameters
        ----------
        loss
            Loss function to be minimized during training.
            By default None.
        optimizer
            Training optimizer.
            By default None
        start_epoch
            First training epoch.
            By default 1
        stop_epoch
            Last training epoch.
            By default 2.
        early_stopping
            If True, then use `val_dataset` for early stopping.
            Ignored if `val_dataset` is `None`.
            By default False.
        patience
            Number of epochs of non-improving validation loss
            before training stops early. Ignored if `early_stopping` is `False`.
            By default 5.
        handlers
            Handlers to run.
            By default None.
        log_filepath
            Filepath to log progress, handler updates, etc.
            By default None.
        **loader_kwargs
            Dataset.loader kwargs.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.early_stopping = early_stopping
        self.patience = patience
        self.loader_kwargs = loader_kwargs
        self.handlers = handlers
        self.log_filepath = log_filepath
        self._state = {}

    def state_dict(self):
        return {
            "loss": self.loss,
            "optimizer": self.optimizer,
            "start_epoch": self.start_epoch,
            "stop_epoch": self.stop_epoch,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "loader_kwargs": self.loader_kwargs,
            "handlers": self.handlers,
            "log_filepath": self.log_filepath,
            # "_state": self._state,
        }

    def load_state_dict(self, state_dict):
        self.loss = state_dict["loss"]
        self.optimizer = state_dict["optimizer"]
        self.start_epoch = state_dict["start_epoch"]
        self.stop_epoch = state_dict["stop_epoch"]
        self.early_stopping = state_dict["early_stopping"]
        self.patience = state_dict["patience"]
        self.loader_kwargs = state_dict["loader_kwargs"]
        self.handlers = state_dict["handlers"]
        self.log_filepath = state_dict["log_filepath"]
        # self._state = state_dict["_state"]

    def fit(
        self,
        model: luz.Module,
        dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> None:
        """Train model.

        Parameters
        ----------
        model
            Model to be trained.
        dataset
            Training data.
        val_dataset
            Validation data, by default None.
        device
            Device to use for training, by default "cpu".
        """
        model.migrate(device)

        # NOTE: must come after migrate
        optimizer = self.optimizer.link(model=model)

        loader = dataset.loader(**self.loader_kwargs)
        self._state = {}
        self.update_state(
            flag=luz.Flag.TRAINING,
            trainer=self,
            model=model,
            optimizer=optimizer,
            loader=loader,
            train_history=[],
            val_history=[],
        )
        if self.early_stopping:
            self.update_state(patience=self.patience)

        self._call_event(luz.Event.TRAINING_STARTED)

        for epoch in range(self.start_epoch, self.stop_epoch):
            self.update_state(epoch=epoch)
            self._call_event(luz.Event.EPOCH_STARTED)

            self.run_epoch(model, loader, device, optimizer)

            if val_dataset is not None:
                self.update_state(flag=luz.Flag.VALIDATING)
                val_loss = self.validate(model, val_dataset, device)
                self.log(f"[Epoch {epoch}] Validation loss: {val_loss}.")
                if self.early_stopping and self._state["patience"] == 0:
                    self.log(f"[Epoch {epoch}]: Stopping early.")
                    break
                self.update_state(flag=luz.Flag.TRAINING)

        self._call_event(luz.Event.TRAINING_ENDED)

    def validate(
        self, model: luz.Module, dataset: luz.Dataset, device: Optional[Device] = "cpu"
    ) -> float:
        """Validate model.

        Parameters
        ----------
        model
            Model to be validated.
        dataset
            Validation data.
        device
            Device to use for validation, by default "cpu".

        Returns
        -------
        float
            Validation loss.
        """
        loader = dataset.loader(**self.loader_kwargs)
        with model.eval():
            val_loss = self.run_epoch(model, loader, device)

        self._state["val_history"].append(val_loss)

        try:
            # FIXME: replace 0.0 with self.delta_thresh?
            if min(self._state["val_history"]) - val_loss < 0.0:
                self._state["patience"] -= 1
            else:
                self._state["patience"] = self.patience
        except ValueError:
            pass

        return val_loss

    def test(
        self,
        model: luz.Module,
        dataset: luz.Dataset,
        device: Optional[Device] = "cpu",
    ) -> float:
        """Test model.

        Parameters
        ----------
        model
            Model to be tested.
        dataset
            Test data.
        device
            Device to use for testing, by default "cpu".
        """
        model.migrate(device)

        loader = dataset.loader(**self.loader_kwargs)
        self._state = {}
        self.update_state(
            flag=luz.Flag.TESTING,
            trainer=self,
            model=model,
            loader=loader,
        )
        self._call_event(luz.Event.TESTING_STARTED)

        self.update_state(epoch=1)
        self._call_event(luz.Event.EPOCH_STARTED)

        test_loss = self.run_epoch(model, loader, device)

        self._call_event(luz.Event.TESTING_ENDED)

        return test_loss

    def run_epoch(
        self,
        model: luz.Module,
        loader: torch.utils.data.DataLoader,
        device: Device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        running_loss = 0.0

        for i, batch in enumerate(loader):
            data = model.get_input(batch)
            target = model.get_target(batch)

            self.update_state(ind=i, data=data, target=target)
            self._call_event(luz.Event.BATCH_STARTED)

            # migrate the input and target tensors to the appropriate device
            data, target = data.to(device), target.to(device)

            if self._state["flag"] == luz.Flag.TRAINING:
                running_loss += model.run_train_batch(data, target, optimizer)
            elif self._state["flag"] == luz.Flag.VALIDATING:
                with model.eval():
                    running_loss += model.run_validate_batch(data, target)
            elif self._state["flag"] == luz.Flag.TESTING:
                with model.eval():
                    running_loss += model.run_test_batch(data, target)
            # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
            # All of the variables defined above are now out of scope!
            # On CPU, they are already deallocated.
            # On GPU, they will be deallocated soon.

            # Make sure deallocation has taken place
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self._call_event(luz.Event.BATCH_ENDED)

        loss = running_loss / len(loader)

        if self._state["flag"] == luz.Flag.TRAINING:
            self._state["train_history"].append(loss)

        self._call_event(luz.Event.EPOCH_ENDED)

        return loss

    def update_state(self, **kwargs: Any) -> None:
        self._state.update(**kwargs)

    def log(self, msg: str) -> None:
        print(msg)

        if self.log_filepath is not None:
            with open(self.log_filepath, "a") as f:
                f.write(f"{msg}\n")

    def _call_event(self, event: luz.Event) -> None:
        for h in self.handlers:
            getattr(h, event.name.lower())(**self._state)
