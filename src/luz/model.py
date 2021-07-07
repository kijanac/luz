from __future__ import annotations
from typing import Callable, Optional, Union

import contextlib
import luz
import pathlib
import torch

__all__ = ["Model"]

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Model(torch.nn.Module):
    def run_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: Optional[Criterion] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Run training algorithm on a single batch.

        Parameters
        ----------
        dataset
            Batch of training data.
        target
            Target tensor.
        criterion
            Training criterion, by default None.
        optimizer
            Training optimizer, by default None.

        Returns
        -------
        torch.Tensor
            Model output.
        float
            Batch loss.
        """
        output = self(data)
        batch_loss = criterion(output, target)

        if optimizer is not None:
            self.backward(batch_loss)
            self.optimizer_step(optimizer)

        return output, batch_loss

    def run_train_batch(self, data, target, criterion, optimizer):
        return self.run_batch(data, target, criterion, optimizer)

    def run_validate_batch(self, data, target, criterion):
        return self.run_batch(data, target, criterion)

    def run_test_batch(self, data, target, criterion):
        return self.run_batch(data, target, criterion)

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

    def save(self, path: Union[str, pathlib.Path]) -> None:
        torch.save(self.state_dict(), path)
        #     {"model": self.state_dict(), "trainer": self.trainer.state_dict()}, path
        # )

    def load(self, path: Union[str, pathlib.Path]):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)  # ["model"])

        # self.trainer = luz.Trainer()
        # self.trainer.load_state_dict(state_dict["trainer"])

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
