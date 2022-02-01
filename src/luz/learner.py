from __future__ import annotations
from typing import Callable, Optional, Union

import luz
import torch

__all__ = ["Learner"]

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Learner:
    def __init__(self) -> None:
        """Learner."""
        self.hparams = None
        self.runners = {
            "preprocess": None,
            "train": None,
            "validate": None,
            "test": None,
        }

    def model(self) -> torch.nn.Module:
        """Get model.

        Returns
        -------
        torch.nn.Module
            Model.
        """
        raise NotImplementedError

    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Get optimizer.

        Parameters
        ----------
        model
            Model.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer.
        """
        raise NotImplementedError

    def criterion(self) -> Criterion:
        """Get loss function.

        Returns
        -------
        Criterion
            Loss function.
        """
        raise NotImplementedError

    def loader(self, dataset: luz.Dataset, stage: str) -> torch.utils.data.DataLoader:
        """Get dataloader.

        Parameters
        ----------
        dataset
            Dataset.
        stage
            Current stage.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader.
        """
        return dataset.loader()

    def transform(self, state: luz.State) -> None:
        """Create and apply transform to runners.

        Parameters
        ----------
        state
            Preprocessor state.
        """
        pass

    def callbacks(self, runner: luz.Runner, stage: str) -> None:
        """Apply callbacks to runners.

        Parameters
        ----------
        runner
            Runner.
        stage
            Current stage.
        """
        pass

    def learn(
        self,
        train_dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        test_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> torch.nn.Module:
        """Learn a model based on a given dataset.

        Parameters
        ----------
        train_dataset
            Training dataset used to learn a model.
        val_dataset
            Validation dataset, by default None.
        test_dataset
            Test dataset, by default None.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        torch.nn.Module
            Learned model.
        """
        model = self.model().to(device)
        optimizer = self.optimizer(model)
        criterion = self.criterion()

        self._runners(model, train_dataset, val_dataset, test_dataset)

        self.runners["train"].RUNNER_STARTED.attach(
            luz.UpdateState(optimizer=optimizer, criterion=criterion)
        )

        if self.runners["preprocess"] is not None:
            self.runners["preprocess"].RUNNER_STARTED.attach(
                luz.UpdateState(criterion=criterion)
            )
            self.runners["preprocess"].RUNNER_ENDED.attach(self._attach_transform)
            self.runners["train"].RUNNER_STARTED.attach(
                luz.Run(self.runners["preprocess"], device)
            )
            self.callbacks(self.runners["preprocess"], "preprocess")

        if val_dataset is not None:
            self.runners["validate"].RUNNER_STARTED.attach(
                luz.UpdateState(criterion=criterion)
            )
            self.runners["train"].EPOCH_ENDED.attach(
                luz.Run(self.runners["validate"], device)
            )
            self.callbacks(self.runners["validate"], "validate")

        if test_dataset is not None:
            self.runners["test"].RUNNER_STARTED.attach(
                luz.UpdateState(criterion=criterion)
            )
            self.runners["train"].RUNNER_ENDED.attach(
                luz.Run(self.runners["test"], device)
            )
            self.callbacks(self.runners["test"], "test")

        self.callbacks(self.runners["train"], "train")

        self.runners["train"].run(device=device)

        if test_dataset is not None:
            return (
                self.runners["train"].state.model,
                self.runners["test"].state.metrics["loss"],
            )
        return self.runners["train"].state.model

    def _runners(
        self,
        model: torch.nn.Module,
        train_dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        test_dataset: Optional[luz.Dataset] = None,
    ) -> None:
        self.runners["train"] = self.runner(model, train_dataset, "train")
        self.runners["preprocess"] = self.runner(model, train_dataset, "preprocess")
        if val_dataset is not None:
            self.runners["validate"] = self.runner(model, val_dataset, "validate")
        if test_dataset is not None:
            self.runners["test"] = self.runner(model, test_dataset, "test")

    def _attach_transform(self, state: luz.State) -> None:
        transform = self.transform(state)
        for k in ["train", "validate", "test"]:
            if self.runners[k] is not None:
                self.runners[k].state.update(transform=transform)
