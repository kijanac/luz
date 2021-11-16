from __future__ import annotations
from typing import Callable, Iterable, Optional, Union

# import argparse
import copy
import luz
import torch

from .predictors import BaseLearner

__all__ = ["Learner"]

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Learner(BaseLearner):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.model = None
        self.score = None

    # def add_args(self, parser):
    #     raise NotImplementedError

    # def parser(self):
    #     parser = argparse.ArgumentParser()
    #     self.add_args(parser)
    #     return parser

    # def parse_args(self):
    #     parser = self.parser()
    #     self.args = parser.parse_args()

    # LOADERS

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

    # TRANSFORM

    def transform(self, dataset):
        pass

    # CALLBACKS

    def callbacks(self) -> Union[luz.Callback, Iterable[luz.Callback]]:
        return []

    def train_callbacks(self) -> Union[luz.Callback, Iterable[luz.Callback]]:
        return self.callbacks()

    def test_callbacks(self) -> Union[luz.Callback, Iterable[luz.Callback]]:
        return self.callbacks()

    def metrics(self):
        return []

    # LOGGERS

    def loggers(self) -> Union[luz.Logger, Iterable[luz.Logger]]:
        return luz.ConsoleLogger()

    def train_loggers(self) -> Union[luz.Logger, Iterable[luz.Logger]]:
        return self.loggers()

    def test_loggers(self) -> Union[luz.Logger, Iterable[luz.Logger]]:
        return self.loggers()

    # NON-CONFIGURABLE METHODS

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
        # try:
        #     self.parse_args()
        # except NotImplementedError:
        #     pass

        # self.reset_state()

        self.model = None
        self.score = None

        transform = self.transform(train_dataset)

        if hasattr(self, "get_input"):
            model = luz.Model(self.nn(), transform, get_input=self.get_input).to(device)
        else:
            model = luz.Model(self.nn(), transform).to(device)

        # NOTE: must come after migrate
        optimizer = self.optimizer(model=model)
        criterion = self.criterion()

        loggers = self.train_loggers()
        if isinstance(loggers, luz.Logger):
            # only 1 logger, so wrap it in a list
            loggers = [loggers]
        callbacks = self.train_callbacks()
        if isinstance(callbacks, luz.Callback):
            # only 1 callback, so wrap it in a list
            callbacks = [callbacks]
        # metrics = self.metrics()

        train_loader = self.train_loader(train_dataset)
        if val_dataset is not None:
            val_loader = self.val_loader(val_dataset)
        else:
            val_loader = None

        params = dict(max_epochs=1, val_every=1, early_stopping=False, patience=5)
        params.update(self.fit_params())

        self.fit(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            val_loader=val_loader,
            loggers=loggers,
            callbacks=callbacks,
            # metrics=metrics,
            device=device,
            **params
        )

        self.model = model

        return model

    def predict(self, model, loader, device):
        model.to(device)

        loggers = self.test_loggers()
        if isinstance(loggers, luz.Logger):
            # only 1 logger, so wrap it in a list
            loggers = [loggers]
        callbacks = self.test_callbacks()
        if isinstance(callbacks, luz.Callback):
            # only 1 callback, so wrap it in a list
            callbacks = [callbacks]
        # FIXME: fix state
        self.run_epoch(model, loader, device, callbacks, state={})
        # FIXME: need to return predictions, not loss

    def evaluate(
        self,
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
        model = self.model

        model.to(device)

        loader = self.test_loader(dataset)

        criterion = self.criterion()

        loggers = self.test_loggers()
        if isinstance(loggers, luz.Logger):
            # only 1 logger, so wrap it in a list
            loggers = [loggers]
        callbacks = self.test_callbacks()
        if isinstance(callbacks, luz.Callback):
            # only 1 callback, so wrap it in a list
            callbacks = [callbacks]

        state = {}
        state.update(
            flag=luz.Flag.TESTING,
            learner=self,
            model=model,
            loader=loader,
            loggers=loggers,
        )
        luz.Event.TESTING_STARTED(callbacks, **state)

        state.update(epoch=1)
        luz.Event.EPOCH_STARTED(callbacks, **state)

        test_loss = self.run_epoch(model, loader, device, callbacks, criterion, state)

        luz.Event.TESTING_ENDED(callbacks, **state)

        self.score = test_loss

        return test_loss

    def fit(
        self,
        model,
        train_loader: torch.utils.data.DataLoader,
        criterion: Criterion,
        optimizer: torch.optim.Optimizer,
        max_epochs=1,
        val_every=1,
        early_stopping=False,
        patience=5,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        loggers: Iterable[luz.Logger] = None,
        callbacks: Iterable[luz.Callback] = None,
        # metrics = None,
        device: Optional[Device] = "cpu",
    ) -> None:
        """Train model.

        Parameters
        ----------
        train_dataset
            Training data.
        val_dataset
            Validation data, by default None.
        device
            Device to use for training, by default "cpu".
        """

        state = {}
        state.update(
            flag=luz.Flag.TRAINING,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=max_epochs,
            val_every=val_every,
            early_stopping=early_stopping,
            patience=patience,
            loader=train_loader,
            train_loader=train_loader,
            train_history=[],
            val_history=[],
            loggers=loggers,
            device=device,
        )
        if val_loader is not None:
            state.update(val_loader=val_loader)
        if early_stopping:
            state.update(best_model=copy.deepcopy(model.state_dict()))

        luz.Event.TRAINING_STARTED(callbacks, **state)

        for epoch in range(max_epochs):
            state.update(epoch=epoch + 1)
            luz.Event.EPOCH_STARTED(callbacks, **state)

            self.run_epoch(
                model,
                train_loader,
                device,
                callbacks,
                # metrics,
                criterion,
                state,
                optimizer,
            )

            # for m in metrics:
            #     m.compute()

            if val_loader is not None and epoch % val_every == 0:
                state.update(flag=luz.Flag.VALIDATING)

                with model.eval():
                    val_loss = self.run_epoch(
                        model,
                        val_loader,
                        device,
                        callbacks,
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

                if early_stopping and state["patience"] == 0:
                    model.load_state_dict(state["best_model"])
                    break

                state.update(flag=luz.Flag.TRAINING)

            luz.Event.EPOCH_ENDED(callbacks, **state)

        luz.Event.TRAINING_ENDED(callbacks, **state)

    def run_epoch(
        self,
        model,
        loader: torch.utils.data.DataLoader,
        device: Device,
        callbacks,
        # metrics,
        criterion: Criterion,
        state,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        running_loss = 0.0

        for i, batch in enumerate(loader):
            data = batch  # self.get_input(batch)
            target = model.get_target(batch)

            state.update(ind=i, data=data, target=target)
            luz.Event.BATCH_STARTED(callbacks, **state)

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

            luz.Event.BATCH_ENDED(callbacks, **state)

        loss = running_loss / len(loader)

        if state["flag"] == luz.Flag.TRAINING:
            state["train_history"].append(loss)

        luz.Event.EPOCH_ENDED(callbacks, **state)
        # for m in metrics:
        #     m.update(**state)

        return loss

    def state_dict(self):
        return {
            "loss": self.params.loss,
            "optimizer": self.params.optimizer,
            "max_epochs": self.params.max_epochs,
            "early_stopping": self.params.early_stopping,
            "patience": self.params.patience,
            # "loader_kwargs": self._fit_params["loader_kwargs"],
            "callbacks": self.callbacks,
            "loggers": self.params.loggers,
            # "_state": self._state,
        }

    def load_state_dict(self, state_dict):
        self.loss = state_dict["loss"]
        self.optimizer = state_dict["optimizer"]
        self.max_epochs = state_dict["max_epochs"]
        self.early_stopping = state_dict["early_stopping"]
        self.patience = state_dict["patience"]
        # self._fit_params["loader_kwargs"] = state_dict["loader_kwargs"]
        self.callbacks = state_dict["callbacks"]
        self.loggers = state_dict["loggers"]
        # self._state = state_dict["_state"]
