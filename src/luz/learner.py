from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Union

from abc import ABC, abstractmethod
import argparse
import copy
import luz
import torch

__all__ = ["Learner"]

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Learner(ABC):
    def add_args(self, parser):
        raise NotImplementedError

    def parser(self):
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        return parser

    def parse_args(self):
        parser = self.parser()
        self.args = parser.parse_args()

    @abstractmethod
    def model(self) -> luz.Model:
        """Return model to be trained.

        Returns
        -------
        luz.Model
            Model to be trained.
        """
        pass

    @abstractmethod
    def criterion(self) -> Criterion:
        pass

    @abstractmethod
    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def fit_params(self) -> dict[str, Any]:
        """Return fit parameters.

        Parameters
        ----------
        train_dataset
            Training dataset used to learn a model.
        val_dataset
            Validation dataset.
        device
            Device to use for learning.

        Returns
        -------
        dict[str, Any]
            Dictionary of fit parameters.
        """
        pass

    def hyperparams(self, tuner: luz.Tuner) -> dict[str, Any]:
        """Specify hyperparameters for tuning.

        Parameters
        ----------
        tuner
            Hyperparameter tuner.

        Returns
        -------
        dict[str, Any]
            Dictionary of hyperparameters to be tuned.
            Each key is the hyperparameter name (used to access the sampled value),
            and values are tuning objects.

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError

    def scorer(self) -> luz.Scorer:
        raise NotImplementedError

    def tuner(self) -> luz.Tuner:
        raise NotImplementedError

    def handlers(self) -> Union[luz.Handler, Iterable[luz.Handler]]:
        return []

    def train_handlers(self) -> Union[luz.Handler, Iterable[luz.Handler]]:
        return self.handlers()

    def test_handlers(self) -> Union[luz.Handler, Iterable[luz.Handler]]:
        return self.handlers()

    def loggers(self) -> Union[luz.Logger, Iterable[luz.Logger]]:
        return luz.ConsoleLogger()

    def train_loggers(self) -> Union[luz.Logger, Iterable[luz.Logger]]:
        return self.loggers()

    def test_loggers(self) -> Union[luz.Logger, Iterable[luz.Logger]]:
        return self.loggers()

    def loader(self, dataset: luz.Dataset) -> torch.utils.data.DataLoader:
        return dataset.loader(
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

    def train_loader(self, dataset: luz.Dataset) -> torch.utils.data.DataLoader:
        return self.loader(dataset)

    def val_loader(self, dataset: luz.Dataset) -> torch.utils.data.DataLoader:
        return self.loader(dataset)

    def test_loader(self, dataset: luz.Dataset) -> torch.utils.data.DataLoader:
        return self.loader(dataset)

    def transform(self, train_dataset: luz.Dataset) -> luz.Transform:
        return None

    # def train_transform(self, train_dataset: luz.Dataset) -> luz.Transform:
    #     return self.transform(train_dataset)

    # def test_transform(self, train_dataset: luz.Dataset) -> luz.Transform:
    #     return self.transform(train_dataset)

    def learn(
        self,
        train_dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> luz.Model:
        """Learn a model based on a given dataset.

        Parameters
        ----------
        train_dataset
            Training dataset used to learn a model.
        val_dataset
            Validation dataset, by default None.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        luz.Model
            Learned model.
        """
        try:
            self.parse_args()
        except NotImplementedError:
            pass

        model = self.model()

        self._transform = self.transform(train_dataset)

        self.fit(model, train_dataset, val_dataset, device)

        return model

    def score(self, dataset, device: Optional[Device] = "cpu") -> luz.Score:
        """Learn a model and estimate its future performance based on a given dataset.

        Parameters
        ----------
        dataset
            Dataset used to learn and score a model.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        luz.Score
            Learned model and estimated performance.
        """
        return self.scorer().score(self, dataset, device)

    def tune(
        self, dataset, device: Optional[Device] = "cpu"
    ) -> tuple[luz.Model, float, dict[str, Any]]:
        """Tune hyperparameters, learn a model with optimal hyperparameters,
        and estimate its future performance based on a given dataset.

        Parameters
        ----------
        dataset
            Dataset used to learn and score a model.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        luz.Model
            Learned model with optimal hyperparameters.
        float
            Estimated future performance of learned model.
        dict[str, Any]
            Optimal hyperparameters.
        """
        tuner = self.tuner()
        self._tuner = tuner
        for exp in tuner.tune(**self.hyperparams(tuner)):
            self.hparams = exp
            model, score = tuner.score(self, dataset, device)

        return (
            tuner.best_model,
            tuner.best_score,
            tuner.best_hyperparameters,
        )

    def fit(
        self,
        model: luz.Module,
        train_dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> None:
        """Train model.

        Parameters
        ----------
        model
            Model to be trained.
        train_dataset
            Training data.
        val_dataset
            Validation data, by default None.
        device
            Device to use for training, by default "cpu".
        """
        if self._transform is not None:
            train_loader = self.train_loader(
                train_dataset.use_transform(self._transform)
            )
            val_loader = self.val_loader(val_dataset.use_transform(self._transform))
        else:
            train_loader = self.train_loader(train_dataset)
            val_loader = self.val_loader(val_dataset)

        params = dict(start_epoch=1, stop_epoch=2, early_stopping=True, patience=5)
        params.update(self.fit_params())
        start_epoch = params["start_epoch"]
        stop_epoch = params["stop_epoch"]
        early_stopping = params["early_stopping"]
        patience = params["patience"]

        model.migrate(device)

        # NOTE: must come after migrate
        optimizer = self.optimizer(model=model)
        criterion = self.criterion()

        loggers = self.train_loggers()
        if isinstance(loggers, luz.Logger):
            # only 1 logger, so wrap it in a list
            loggers = [loggers]
        handlers = self.train_handlers()
        if isinstance(handlers, luz.Handler):
            # only 1 handler, so wrap it in a list
            handlers = [handlers]

        state = {}
        state.update(
            flag=luz.Flag.TRAINING,
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            train_history=[],
            val_history=[],
            loggers=loggers,
        )
        if early_stopping:
            state.update(
                patience=patience, best_model=copy.deepcopy(model.state_dict())
            )

        luz.Event.TRAINING_STARTED(handlers, **state)

        for epoch in range(start_epoch, stop_epoch + 1):
            state.update(epoch=epoch)
            luz.Event.EPOCH_STARTED(handlers, **state)

            self.run_epoch(
                model, train_loader, device, handlers, criterion, state, optimizer
            )

            if val_dataset is not None:
                state.update(flag=luz.Flag.VALIDATING)

                # val_loss = self.validate(model, val_loader, device)

                with model.eval():
                    val_loss = self.run_epoch(
                        model,
                        val_loader,
                        device,
                        handlers,
                        criterion,
                        state,
                    )

                    state["val_history"].append(val_loss)

                    try:
                        # FIXME: replace 0.0 with self.delta_thresh?
                        if min(state["val_history"]) - val_loss < 0.0:
                            state["patience"] -= 1
                        else:
                            state.update(
                                patience=patience,
                                best_model=copy.deepcopy(model.state_dict()),
                            )
                    except ValueError:
                        pass

                # return val_loss

                # self.log(f"[Epoch {epoch}] Validation loss: {val_loss}.")

                if early_stopping and state["patience"] == 0:
                    # self.log(f"[Epoch {epoch}]: Stopping early.")
                    model.load_state_dict(state["best_model"])
                    break

                state.update(flag=luz.Flag.TRAINING)

        luz.Event.TRAINING_ENDED(handlers, **state)

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

        if self._transform is not None:
            loader = self.test_loader(dataset.use_transform(self._transform))
        else:
            loader = self.test_loader(dataset)

        criterion = self.criterion()

        loggers = self.test_loggers()
        if isinstance(loggers, luz.Logger):
            # only 1 logger, so wrap it in a list
            loggers = [loggers]
        handlers = self.test_handlers()
        if isinstance(handlers, luz.Handler):
            # only 1 handler, so wrap it in a list
            handlers = [handlers]

        state = {}
        state.update(
            flag=luz.Flag.TESTING,
            trainer=self,
            model=model,
            loader=loader,
            loggers=loggers,
        )
        luz.Event.TESTING_STARTED(handlers, **state)

        state.update(epoch=1)
        luz.Event.EPOCH_STARTED(handlers, **state)

        test_loss = self.run_epoch(model, loader, device, handlers, criterion, state)

        luz.Event.TESTING_ENDED(handlers, **state)

        return test_loss

    def run_epoch(
        self,
        model: luz.Module,
        loader: torch.utils.data.DataLoader,
        device: Device,
        handlers,
        criterion: Criterion,
        state,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        running_loss = 0.0

        for i, batch in enumerate(loader):
            data = model.get_input(batch)
            target = model.get_target(batch)

            state.update(ind=i, data=data, target=target)
            luz.Event.BATCH_STARTED(handlers, **state)

            # migrate the input and target tensors to the appropriate device
            data, target = data.to(device), target.to(device)

            if state["flag"] == luz.Flag.TRAINING:
                output, batch_loss = model.run_train_batch(
                    data, target, criterion, optimizer
                )
            elif state["flag"] == luz.Flag.VALIDATING:
                with model.eval():
                    output, batch_loss = model.run_validate_batch(
                        data, target, criterion
                    )
            elif state["flag"] == luz.Flag.TESTING:
                with model.eval():
                    output, batch_loss = model.run_test_batch(data, target, criterion)

            running_loss += batch_loss.item()

            state.update(output=output, loss=batch_loss)

            # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
            # All of the variables defined above are now out of scope!
            # On CPU, they are already deallocated.
            # On GPU, they will be deallocated soon.

            # Make sure deallocation has taken place
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            luz.Event.BATCH_ENDED(handlers, **state)

        loss = running_loss / len(loader)

        if state["flag"] == luz.Flag.TRAINING:
            state["train_history"].append(loss)

        luz.Event.EPOCH_ENDED(handlers, **state)

        return loss

    def state_dict(self):
        return {
            "loss": self.params.loss,
            "optimizer": self.params.optimizer,
            "start_epoch": self.params.start_epoch,
            "stop_epoch": self.params.stop_epoch,
            "early_stopping": self.params.early_stopping,
            "patience": self.params.patience,
            # "loader_kwargs": self._fit_params["loader_kwargs"],
            "handlers": self.handlers,
            "loggers": self.params.loggers,
            # "_state": self._state,
        }

    def load_state_dict(self, state_dict):
        self.loss = state_dict["loss"]
        self.optimizer = state_dict["optimizer"]
        self.start_epoch = state_dict["start_epoch"]
        self.stop_epoch = state_dict["stop_epoch"]
        self.early_stopping = state_dict["early_stopping"]
        self.patience = state_dict["patience"]
        # self._fit_params["loader_kwargs"] = state_dict["loader_kwargs"]
        self.handlers = state_dict["handlers"]
        self.loggers = state_dict["loggers"]
        # self._state = state_dict["_state"]
