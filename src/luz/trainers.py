from __future__ import annotations
from typing import Callable, Iterable, Optional, Tuple, Union

__all__ = ["Trainer", "SupervisedTrainer"]

import luz
import torch


class Trainer:
    def __init__(
        self,
        loss=None,
        optimizer: Optional[luz.Optimizer] = None,
        start_epoch: Optional[int] = 1,
        stop_epoch: Optional[int] = 2,
        handlers: Optional[Iterable[luz.Handler]] = None,
        process_batch: Optional[
            Callable[luz.Data, Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
        **loader_kwargs: Union[int, bool, luz.Transform],
    ) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.handlers = tuple(handlers or [])
        self.process_batch = process_batch or (lambda batch: (batch.x, batch.y))
        self.loader_kwargs = loader_kwargs

        self.flag = None
        self.state = {}

    def _call_event(self, event):
        for handler in self.handlers:
            getattr(handler, event.name.lower())(**self.state)

    def process_batch(
        self, batch: luz.Data
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def run_batch(
        self,
        predictor: luz.Predictor,
        data: torch.Tensor,
        target: torch.Tensor,
        device: Union[str, torch.device],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        raise NotImplementedError

    def run(
        self,
        predictor: luz.Predictor,
        dataset: luz.Dataset,
        device: Union[str, torch.device],
        train: bool,
        val_dataset: Optional[luz.Dataset] = None,
    ) -> None:
        self.set_mode(predictor, train)
        self.migrate(predictor, device)

        if train:
            # NOTE: must come after migrate
            optimizer = self.optimizer.link(predictor=predictor)

        loader = dataset.loader(**self.loader_kwargs)

        if train:
            self.state = dict(
                flag=luz.Flag.TRAINING,
                trainer=self,
                predictor=predictor,
                optimizer=optimizer,
                loader=loader,
            )
            self._call_event(event=luz.Event.TRAINING_STARTED)
        else:
            self.state = dict(
                flag=luz.Flag.TESTING,
                trainer=self,
                predictor=predictor,
                loader=loader,
            )
            self._call_event(event=luz.Event.TESTING_STARTED)

        if train:
            start, stop = self.start_epoch, self.stop_epoch
        else:
            start, stop = 1, 2

        for epoch in range(start, stop):
            running_loss = 0.0

            self.state.update(epoch=epoch)
            self._call_event(event=luz.Event.EPOCH_STARTED)

            for i, batch in enumerate(loader):
                data, target = self.process_batch(batch)
                self.state.update(ind=i, data=data, target=target)
                self._call_event(event=luz.Event.BATCH_STARTED)

                # migrate the input and target tensors to the appropriate device
                data, target = data.to(device), target.to(device)

                if train:
                    self.run_batch(predictor, data, target, device, optimizer)
                    if val_dataset is not None:
                        # FIXME: implement this!
                        raise NotImplementedError
                else:
                    running_loss += self.run_batch(predictor, data, target, device)
                # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
                # All of the variables defined above are now out of scope!
                # On CPU, they are already deallocated.
                # On GPU, they will be deallocated soon.

                # Make sure deallocation has taken place
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                self._call_event(event=luz.Event.BATCH_ENDED)
            self._call_event(event=luz.Event.EPOCH_ENDED)

        if train:
            self._call_event(event=luz.Event.TRAINING_ENDED)
        else:
            self._call_event(event=luz.Event.TESTING_ENDED)

            return running_loss / len(loader)

    def set_mode(self, predictor: luz.Predictor, train: bool) -> None:
        predictor.train() if train else predictor.eval()

    def migrate(
        self, predictor: luz.Predictor, device: Union[str, torch.device]
    ) -> None:
        predictor.to(device=device)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()


class SupervisedTrainer(Trainer):
    def run_batch(
        self,
        predictor: luz.Predictor,
        data: torch.Tensor,
        target: torch.Tensor,
        device: Union[str, torch.device],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        output = predictor(data)
        loss = self.loss(output, target)

        if optimizer is not None:
            self.backward(loss)
            self.optimizer_step(optimizer)

        self.state.update(output=output, loss=loss)

        return loss.item()


class SemisupervisedTrainer(Trainer):
    pass


class UnupervisedTrainer(Trainer):
    pass
