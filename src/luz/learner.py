from __future__ import annotations
from typing import Callable, Optional, Union

import luz
import torch

__all__ = ["Learner"]

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Learner:
    def __init__(self) -> None:
        self.hparams = None
        self.preprocessor = None
        self.trainer = None
        self.validator = None
        self.tester = None

    def model(self) -> torch.nn.Module:
        """Get model."""
        raise NotImplementedError

    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Get optimizer."""
        raise NotImplementedError

    def criterion(self) -> Criterion:
        """Get loss function."""
        raise NotImplementedError

    def loader(self, dataset: luz.Dataset, stage: str) -> torch.utils.data.DataLoader:
        """Get dataloader."""
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
        """Apply callbacks to runners."""
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

        self.trainer.RUNNER_STARTED.attach(
            luz.UpdateState(optimizer=optimizer, criterion=criterion)
        )

        if self.preprocessor is not None:
            self.preprocessor.RUNNER_STARTED.attach(
                luz.UpdateState(criterion=criterion)
            )
            self.preprocessor.RUNNER_ENDED.attach(self._attach_transform)
            self.trainer.RUNNER_STARTED.attach(luz.Run(self.preprocessor, device))
            self.callbacks(self.preprocessor, "preprocess")

        if val_dataset is not None:
            self.validator.RUNNER_STARTED.attach(luz.UpdateState(criterion=criterion))
            self.trainer.EPOCH_ENDED.attach(luz.Run(self.validator, device))
            self.callbacks(self.validator, "validate")

        if test_dataset is not None:
            self.tester.RUNNER_STARTED.attach(luz.UpdateState(criterion=criterion))
            self.trainer.RUNNER_ENDED.attach(luz.Run(self.tester, device))
            self.callbacks(self.tester, "test")

        self.callbacks(self.trainer, "train")

        self.trainer.run(device=device)

        if test_dataset is not None:
            return self.trainer.state.model, self.tester.state.metrics["loss"]
        return self.trainer.state.model

    def _runners(
        self,
        model: torch.nn.Module,
        train_dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        test_dataset: Optional[luz.Dataset] = None,
    ) -> None:
        self.trainer = self.runner(model, train_dataset, "train")
        self.preprocessor = self.runner(model, train_dataset, "preprocess")
        if val_dataset is not None:
            self.validator = self.runner(model, val_dataset, "validate")
        if test_dataset is not None:
            self.tester = self.runner(model, test_dataset, "test")

    def _attach_transform(self, state: luz.State) -> None:
        transform = self.transform(state)
        for runner in [self.trainer, self.validator, self.tester]:
            if runner is not None:
                runner.state.update(transform=transform)
