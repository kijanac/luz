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
        **loader_kwargs: Union[int, bool, luz.Transform],
    ) -> None:
        """Algorithm to train a model using data.

        Parameters
        ----------
        loss
            Loss function to be minimized during training, by default None
        optimizer
            Training optimizer, by default None
        start_epoch
            First training epoch, by default 1
        stop_epoch
            Last training epoch, by default 2
        early_stopping
            If True, then use `val_dataset` for early stopping; by default False.
            Ignored if `val_dataset` is `None`.
        patience
            Number of epochs of non-improving validation loss
            before training stops early; by default 5.
            Ignored if `early_stopping` is `False`.
        handlers
            Handlers to run, by default None.
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
        self.log(
            flag=luz.Flag.TRAINING,
            trainer=self,
            model=model,
            optimizer=optimizer,
            loader=loader,
            train_history=[],
            val_history=[],
        )
        if self.early_stopping:
            self.log(patience=self.patience)

        self._call_event(luz.Event.TRAINING_STARTED)

        for epoch in range(self.start_epoch, self.stop_epoch):
            self.log(epoch=epoch)
            self._call_event(luz.Event.EPOCH_STARTED)

            self.run_epoch(model, loader, device, True, optimizer)

            if val_dataset is not None:
                self.log(flag=luz.Flag.VALIDATING)
                val_loss = self.validate(model, val_dataset, device)
                print(f"[Epoch {epoch}] Validation loss: {val_loss}.")
                if self.early_stopping and self._state["patience"] == 0:
                    print(f"[Epoch {epoch}]: Stopping early.")
                    break
                self.log(flag=luz.Flag.TRAINING)

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
            val_loss = self.run_epoch(model, loader, device, train=False)

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
        self.log(
            flag=luz.Flag.TESTING,
            trainer=self,
            model=model,
            loader=loader,
        )
        self._call_event(luz.Event.TESTING_STARTED)

        self.log(epoch=1)
        self._call_event(luz.Event.EPOCH_STARTED)

        test_loss = self.run_epoch(model, loader, device, train=False)

        self._call_event(luz.Event.TESTING_ENDED)

        return test_loss

    def run_epoch(
        self,
        model: luz.Module,
        loader: torch.utils.data.DataLoader,
        device: Device,
        train: bool,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        running_loss = 0.0

        for i, batch in enumerate(loader):
            data = model.get_input(batch)
            target = model.get_target(batch)

            self.log(ind=i, data=data, target=target)
            self._call_event(luz.Event.BATCH_STARTED)

            # migrate the input and target tensors to the appropriate device
            data, target = data.to(device), target.to(device)

            if train:
                running_loss += model.run_batch(data, target, device, optimizer)
            else:
                with model.eval():
                    running_loss += model.run_batch(data, target, device)
            # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
            # All of the variables defined above are now out of scope!
            # On CPU, they are already deallocated.
            # On GPU, they will be deallocated soon.

            # Make sure deallocation has taken place
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self._call_event(luz.Event.BATCH_ENDED)

        loss = running_loss / len(loader)

        if train:
            self._state["train_history"].append(loss)

        self._call_event(luz.Event.EPOCH_ENDED)

        return loss

    def log(self, **kwargs: Any) -> None:
        self._state.update(**kwargs)

    def _call_event(self, event: luz.Event) -> None:
        for h in self.handlers:
            getattr(h, event.name.lower())(**self._state)
