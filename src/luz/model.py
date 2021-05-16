from __future__ import annotations
from typing import Callable, Optional, Union

import contextlib
import luz
import pathlib
import torch

__all__ = ["Model"]

Device = Union[str, torch.device]
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Model(torch.nn.Module):
    def run_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Run training algorithm on a single batch.

        Parameters
        ----------
        dataset
            Batch of training data.
        target
            Target tensor.
        optimizer
            Training optimizer, by default None.

        Returns
        -------
        float
            Batch loss.
        """
        output = self(data)
        loss = self.loss(output, target)

        if optimizer is not None:
            self.backward(loss)
            self.optimizer_step(optimizer)

        self.trainer.update_state(output=output, loss=loss)

        return loss.item()

    def run_train_batch(self, data, target, optimizer):
        return self.run_batch(data, target, optimizer)

    def run_validate_batch(self, data, target):
        return self.run_batch(data, target)

    def run_test_batch(self, data, target):
        return self.run_batch(data, target)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate loss.

        Parameters
        ----------
        loss
            Loss tensor.
        """
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step training optimizer.

        Parameters
        ----------
        optimizer
            Training optimizer.
        """
        optimizer.step()
        optimizer.zero_grad()

    def get_input(self, batch: luz.Data) -> torch.Tensor:
        """Get input from batched data.

        Parameters
        ----------
        batch
            Batched data.

        Returns
        -------
        torch.Tensor
            Input tensor.
        """
        return batch.x

    def get_target(self, batch: luz.Data) -> Optional[torch.Tensor]:
        """Get target from batched data.

        Parameters
        ----------
        batch
            Batched data.

        Returns
        -------
        Optional[torch.Tensor]
            Target tensor.
        """
        return batch.y

    # NON-CONFIGURABLE METHODS BELOW

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def loss(self) -> Loss:
        return self.trainer.loss

    def save(self, path: Union[str, pathlib.Path]) -> None:
        torch.save(
            {"model": self.state_dict(), "trainer": self.trainer.state_dict()}, path
        )

    def load(self, path: Union[str, pathlib.Path]):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict["model"])

        self.trainer = luz.Trainer()
        self.trainer.load_state_dict(state_dict["trainer"])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass in eval mode.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        with self.eval():
            if hasattr(self, "transform"):
                return self.transform(self.__call__(x))
            return self.__call__(x)

    @contextlib.contextmanager
    def eval(self, no_grad: Optional[bool] = True) -> None:
        """Context manager to operate in eval mode.

        Parameters
        ----------
        no_grad
            If True use torch.no_grad(), by default True.
        """
        training = True if self.training else False

        nc = contextlib.nullcontext()
        with torch.no_grad() if no_grad else nc:
            try:
                if training:
                    super().eval()
                yield
            finally:
                if training:
                    self.train()

    def migrate(self, device: Device) -> None:
        self.to(device=device)

    def log(self, msg: str) -> None:
        self.trainer.log(msg)

    # def call_event(self, event: luz.Event, **kwargs: Any) -> None:
    #     for h in self.trainer.handlers:
    #         getattr(h, event.name.lower())(model=self, **kwargs)

    def fit(
        self,
        dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> luz.Module:
        """Fit model.

        Parameters
        ----------
        dataset
            Training data.
        val_dataset
            Validation data, by default None.
        device
            Device to use for training, by default "cpu".

        Returns
        -------
        luz.Module
            Trained model.
        """
        self.trainer.fit(self, dataset, val_dataset, device)

        return self

    def validate(self, dataset: luz.Dataset, device: Optional[Device] = "cpu") -> float:
        """Validate model.

        Parameters
        ----------
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
        with self.eval():
            val_loss = self.run_epoch(loader, device, train=False)

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
        dataset: luz.Dataset,
        device: Optional[Device] = "cpu",
    ) -> float:
        return self.trainer.test(self, dataset, device)

    def use_fit_params(self, **kwargs) -> None:
        self.trainer = luz.Trainer(**kwargs)

    # def transform_inputs(self, *args, **kwargs):
    #     return

    def use_transform(self, transform: luz.TensorTransform) -> None:
        """Use transform.

        Parameters
        ----------
        transform
            Tensor transform applied to output during inference.
        """
        self.transform = transform
