from __future__ import annotations
from typing import Callable, Optional, Tuple, Union

from abc import ABC, abstractmethod
import luz
import torch

__all__ = ["Trainer", "SupervisedTrainer"]


Device = Union[str, torch.device]
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ProcessBatch = Callable[[luz.Data], Tuple[torch.Tensor, Optional[torch.Tensor]]]


class Trainer(ABC):
    def __init__(
        self,
        loss: Optional[Loss] = None,
        optimizer: Optional[luz.Optimizer] = None,
        start_epoch: Optional[int] = 1,
        stop_epoch: Optional[int] = 2,
        early_stopping: Optional[bool] = False,
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
        self.handlers = []

        self._state = {}

    def _call_event(self, event: luz.Event) -> None:
        for h in self.handlers:
            getattr(h, event.name.lower())(**self._state)

    def use_process(self, process_batch: ProcessBatch) -> None:
        """Set function to process each batch.

        Parameters
        ----------
        process_batch
            Function to get input and optionally target from data.
        """
        self._process_batch = process_batch

    def use_handlers(self, *handlers: luz.Handler) -> None:
        """Set handlers.

        *handlers
            Handlers to run.
        """
        self.handlers = handlers

    def run(
        self,
        model: luz.Module,
        dataset: luz.Dataset,
        device: Device,
        train: bool,
        val_dataset: Optional[luz.Dataset] = None,
    ) -> None:
        """Run training algorithm.

        Parameters
        ----------
        model
            Model to be trained.
        dataset
            Training data.
        device
            Device to use for training.
        train
            If True, then train, else test.
        val_dataset
            Validation data, by default None.

        """
        self.migrate(model, device)

        if train:
            # NOTE: must come after migrate
            optimizer = self.optimizer.link(model=model)

        loader = dataset.loader(**self.loader_kwargs)

        if train:
            self._state = dict(
                flag=luz.Flag.TRAINING,
                trainer=self,
                model=model,
                optimizer=optimizer,
                loader=loader,
                train_history=[],
                val_history=[],
            )
            if self.early_stopping:
                self._state.update(patience=5)

            self._call_event(luz.Event.TRAINING_STARTED)
        else:
            self._state = dict(
                flag=luz.Flag.TESTING,
                trainer=self,
                model=model,
                loader=loader,
            )
            self._call_event(luz.Event.TESTING_STARTED)

        if train:
            start, stop = self.start_epoch, self.stop_epoch
        else:
            start, stop = 1, 2

        for epoch in range(start, stop):
            running_loss = 0.0

            self._state.update(epoch=epoch)
            self._call_event(luz.Event.EPOCH_STARTED)

            for i, batch in enumerate(loader):
                data, target = self._process_batch(batch)
                self._state.update(ind=i, data=data, target=target)
                self._call_event(luz.Event.BATCH_STARTED)

                # migrate the input and target tensors to the appropriate device
                data, target = data.to(device), target.to(device)

                if train:
                    self.run_batch(model, data, target, device, optimizer)
                else:
                    with model.eval():
                        running_loss += self.run_batch(model, data, target, device)
                # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
                # All of the variables defined above are now out of scope!
                # On CPU, they are already deallocated.
                # On GPU, they will be deallocated soon.

                # Make sure deallocation has taken place
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                self._call_event(luz.Event.BATCH_ENDED)

            if train:
                self._state["train_history"].append(running_loss)

            self._call_event(luz.Event.EPOCH_ENDED)

            if train and val_dataset is not None:
                val_loss = 0.0
                with model.eval():
                    for batch in val_dataset.loader(**self.loader_kwargs):
                        data, target = self._process_batch(batch)
                        # migrate the input and target tensors to the appropriate device
                        data, target = data.to(device), target.to(device)
                        val_loss += self.run_batch(model, data, target, device)

                print(f"[Epoch {epoch}] Validation loss: {val_loss}.")
                if self.early_stopping:
                    try:
                        best_val_loss = min(self._state["val_history"])

                        if best_val_loss - val_loss < 0.0:  # delta_thresh
                            self._state["patience"] -= 1
                        else:
                            self._state["patience"] = self.patience

                        if self._state["patience"] == 0:
                            print(f"[Epoch {epoch}]: Stopping early.")
                            break
                    except ValueError:
                        pass

                self._state["val_history"].append(val_loss)

        if train:
            self._call_event(luz.Event.TRAINING_ENDED)
        else:
            self._call_event(luz.Event.TESTING_ENDED)

            return running_loss / len(loader)

    def migrate(self, model: luz.Module, device: Device) -> None:
        model.to(device=device)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()

    @abstractmethod
    def run_batch(
        self,
        model: luz.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        device: Device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        pass


class SupervisedTrainer(Trainer):
    def _process_batch(
        self, batch: luz.Data
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get output and optionally target from batched data.

        Parameters
        ----------
        batch
            Batched data.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Output tensor and optional target tensor.
        """
        return batch.x, batch.y

    def run_batch(
        self,
        model: luz.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        device: Device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Run training algorithm on a single batch.

        Parameters
        ----------
        model
            Model to be trained.
        dataset
            Batch of training data.
        target
            Target tensor.
        device
            Device to use for training.
        optimizer
            Optimizer, by default None.

        Returns
        -------
        float
            Batch loss.
        """
        output = model(data)
        loss = self.loss(output, target)

        if optimizer is not None:
            self.backward(loss)
            self.optimizer_step(optimizer)

        self._state.update(output=output, loss=loss)

        return loss.item()


class SemisupervisedTrainer(Trainer):
    pass


class UnupervisedTrainer(Trainer):
    pass
