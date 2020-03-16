from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple, Union

# FIXME: shouldn't need to have special graph trainers going forward... solution is to just force every supervised trainer to take in separate x and y tensors! 
__all__ = ["Trainer", "SupervisedTrainer", "SupervisedGraphTrainer"]

import luz
import torch


class Trainer:
    def __init__(
        self,
        loss = None,
        optimizer: Optional[luz.Optimizer] = None,
        start_epoch: Optional[int] = 1,
        stop_epoch: Optional[int] = 1,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = False,
        handlers: Optional[Iterable[luz.Handler]] = None,
        data_transform: Optional[luz.Transform] = None,
        target_transform: Optional[luz.Transform] = None,
        ) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.handlers = tuple(handlers or [])
        self.data_transform = data_transform
        self.target_transform = target_transform

        self.flag = None
        self.state = {}

    def _call_event(self, event):
        for handler in self.handlers:
            getattr(handler, event.name.lower())(**self.state)

    def process_batch(self, batch) -> Tuple[torch.Tensor,Optional[torch.Tensor]]:
        raise NotImplementedError

    def run_batch(self, predictor: luz.Predictor, data: torch.Tensor, target: torch.Tensor, device: Union[str, torch.device], optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        raise NotImplementedError

    def run(
        self,
        predictor: luz.Predictor,
        dataset: luz.Dataset,
        device: Union[str, torch.device],
        train: bool,
    ) -> None:
        if train:
            predictor.model.train().to(device=device)
        else:
            predictor.model.eval().to(device=device)

        # FIXME: this is a hacky way to give model a device so that it can migrate tensors to the same device - fix this
        # p.nn.device = self.device

        if train:
            # NOTE: must come after model.to
            optimizer = self.optimizer.link(predictor=predictor)

        loader = dataset.loader(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

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
                flag=luz.Flag.TESTING, trainer=self, predictor=predictor, loader=loader,
            )
            self._call_event(event=luz.Event.TESTING_STARTED)


        for epoch in range(self.start_epoch if train else 1, self.stop_epoch + 1 if train else 2):
            running_loss = 0.0
            self.state.update(epoch=epoch)
            self._call_event(event=luz.Event.EPOCH_STARTED)
            for i, batch in enumerate(loader):
                data,target = self.process_batch(batch)
                self.state.update(ind=i, data=data, target=target)
                self._call_event(event=luz.Event.BATCH_STARTED)

                if self.data_transform is not None:
                    data = self.data_transform(data)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                    
                # migrate the input and target tensors to the appropriate device
                data, target = data.to(device), target.to(device)

                if train:
                    self.run_batch(predictor,data,target,device,optimizer)
                else:
                    running_loss += self.run_batch(predictor,data,target,device)
                # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
                # All of the variables defined above are now out of scope!
                # On CPU, they are already deallocated. On GPU, they will be deallocated soon.

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

    # def test(
    #     self,
    #     predictor: luz.Predictor,
    #     dataset: luz.Dataset,
    #     device: Union[str, torch.device],
    # ) -> None:
    #     running_loss = 0.0
    #     predictor.model.eval().to(device=device)
    #     # FIXME: this is a hacky way to give model a device so that it can migrate tensors to the same device - fix this
    #     # p.nn.device = self.device

    #     loader = dataset.loader(
    #         batch_size=self.batch_size,
    #         shuffle=self.shuffle,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #     )

    #     self.state = dict(
    #         flag=luz.Flag.TESTING, trainer=self, predictor=predictor, loader=loader,
    #     )
    #     self._call_event(event=luz.Event.TESTING_STARTED)

    #     self.state.update(epoch=1)
    #     self._call_event(event=luz.Event.EPOCH_STARTED)
    #     for i, (data, target) in enumerate(loader):
    #         self.state.update(ind=i, data=data, target=target)
    #         self._call_event(event=luz.Event.BATCH_STARTED)
    #         # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
    #         running_loss += self.run_batch(predictor,data,target,device)
    #         # All of the variables defined above are now out of scope!
    #         # On CPU, they are already deallocated. On GPU, they will be deallocated soon.

    #         # Make sure deallocation has taken place
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         self._call_event(event=luz.Event.BATCH_ENDED)
    #     self._call_event(event=luz.Event.EPOCH_ENDED)

    #     self._call_event(event=luz.Event.TESTING_ENDED)

    #     return running_loss / len(loader)

class SupervisedTrainer(Trainer):
    def process_batch(self, batch):
        return batch.x,batch.y

    def run_batch(self, predictor: luz.Predictor, data: torch.Tensor, target: torch.Tensor, device: Union[str, torch.device], optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        output = predictor(data)
        loss = self.loss(output, target)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.state.update(output=output, loss=loss)

        return loss.item()

class SupervisedGraphTrainer(Trainer):
    def process_batch(self, batch):
        return batch,batch.y

    def run_batch(self, predictor: luz.Predictor, data: torch.Tensor, target: torch.Tensor, device: Union[str, torch.device], optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        output = predictor(data)
        loss = self.loss(output, target)#@torch.Tensor([[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0]]))

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.state.update(output=output, loss=loss)

        return loss.item()

# class SupervisedGraphTrainer:
#     def __init__(
#         self,
#         loss,
#         optimizer: luz.Optimizer,
#         start_epoch: Optional[int] = 1,
#         stop_epoch: Optional[int] = 1,
#         batch_size: Optional[int] = 1,
#         shuffle: Optional[bool] = True,
#         num_workers: Optional[int] = 1,
#         pin_memory: Optional[bool] = False,
#         handlers: Optional[Iterable[luz.Handler]] = None,
#         transform: Optional[luz.Transform] = None,
#     ) -> None:
#         self.flag = None
#         self.handlers = tuple(handlers or [])
#         self.state = {}
#         self.loss = loss
#         self.optimizer = optimizer
#         self.start_epoch = start_epoch
#         self.stop_epoch = stop_epoch
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.transform = transform
        
#     def _call_event(self, event):
#         # Event.TRAINING_STARTED -> 'Event.TRAINING_STARTED' -> 'event.training_started' -> 'training_started'
#         _, event_str = str(event).lower().rsplit(".")
#         for handler in self.handlers:
#             #if hasattr(handler, event_str):
#             getattr(handler, event_str)(**self.state)

#     def train(
#         self,
#         predictor: luz.Predictor,
#         dataset: luz.Dataset,
#         device: Union[str, torch.device],
#     ) -> None:
#         predictor.model.train().to(device=device)
#         # FIXME: this is a hacky way to give model a device so that it can migrate tensors to the same device - fix this
#         # p.nn.device = self.device

#         # NOTE: must come after model.to
#         optimizer = self.optimizer.link(predictor=predictor)

#         loader = dataset.loader(
#             batch_size=self.batch_size,
#             shuffle=self.shuffle,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#         self.state = dict(
#             flag=luz.Flag.TRAINING,
#             trainer=self,
#             predictor=predictor,
#             optimizer=optimizer,
#             loader=loader,
#         )
#         self._call_event(event=luz.Event.TRAINING_STARTED)

#         for epoch in range(self.start_epoch, self.stop_epoch + 1):
#             self.state.update(epoch=epoch)
#             self._call_event(event=luz.Event.EPOCH_STARTED)
#             for i, data in enumerate(loader):
#                 self.state.update(ind=i, data=data)
#                 self._call_event(event=luz.Event.BATCH_STARTED)
#                 # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
#                 self.run_batch(predictor,data,device,optimizer)
#                 # All of the variables defined above are now out of scope!
#                 # On CPU, they are already deallocated. On GPU, they will be deallocated soon.

#                 # Make sure deallocation has taken place
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 self._call_event(event=luz.Event.BATCH_ENDED)
#             self._call_event(event=luz.Event.EPOCH_ENDED)

#         self._call_event(event=luz.Event.TRAINING_ENDED)

#     def test(
#         self,
#         predictor: luz.Predictor,
#         dataset: luz.Dataset,
#         device: Union[str, torch.device],
#     ) -> None:
#         running_loss = 0.0
#         predictor.model.eval().to(device=device)
#         # FIXME: this is a hacky way to give model a device so that it can migrate tensors to the same device - fix this
#         # p.nn.device = self.device

#         loader = dataset.loader(
#             batch_size=self.batch_size,
#             shuffle=self.shuffle,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#         self.state = dict(
#             flag=luz.Flag.TESTING, trainer=self, predictor=predictor, loader=loader,
#         )
#         self._call_event(event=luz.Event.TESTING_STARTED)

#         self.state.update(epoch=1)
#         self._call_event(event=luz.Event.EPOCH_STARTED)
#         for i, (data, target) in enumerate(loader):
#             self.state.update(ind=i, data=data, target=target)
#             self._call_event(event=luz.Event.BATCH_STARTED)
#             # from https://coolnesss.github.io/2019-02-05/pytorch-gotchas
#             running_loss += self.run_batch(predictor,data,target,device)
#             # All of the variables defined above are now out of scope!
#             # On CPU, they are already deallocated. On GPU, they will be deallocated soon.

#             # Make sure deallocation has taken place
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             self._call_event(event=luz.Event.BATCH_ENDED)
#         self._call_event(event=luz.Event.EPOCH_ENDED)

#         self._call_event(event=luz.Event.TESTING_ENDED)

#         return running_loss / len(loader)

#     def run_batch(self, predictor: luz.Predictor, data: torch_geometric.utils.data.Data, device: Union[str, torch.device], optimizer: Optional[torch.optim.Optimizer] = None) -> float:
#         if self.transform is not None:
#             data = self.transform(data)
#         # migrate the input and target tensors to the appropriate device
#         data = data.to(device)

#         output = predictor(data)
#         loss = self.loss(output, data.y@torch.Tensor([[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0]]))

#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         self.state.update(output=output, loss=loss)

#         return loss.item()

class SemisupervisedTrainer(Trainer):
    pass

class UnupervisedTrainer(Trainer):
    pass

    # @classmethod
    # def builder(cls, *args, **kwargs):
    #     def f(*new_args,**new_kwargs):
    #         return cls(*args,*new_args,**kwargs,**new_kwargs)
    #     return f

    # def state_dict(self):
    #     return {'loss': self.loss, 'optimizer': self.cur_optimizer.state_dict(),
    #                    'event_handlers': self.event_handlers,
    #                    'num_epochs': self.num_epochs, 'batch_size': self.batch_size,
    #                    'shuffle': self.shuffle, 'device': self.device}

    # def load_state_dict(self, state_dict):
    #     self.loss = state_dict['loss']
    #     self.cur_optimizer.load_state_dict(state_dict['optimizer'])
    #     self.event_handlers = state_dict['event_handlers']
    #     self.num_epochs = state_dict['num_epochs']
    #     self.batch_size = state_dict['batch_size']
    #     self.shuffle = state_dict['shuffle']
    #     self.device = state_dict['device']

    # def call_event(self, event, state):
    #     #Event.TRAINING_STARTED -> 'Event.TRAINING_STARTED' -> 'event.training_started' -> 'training_started'
    #     _,event_str = str(event).lower().rsplit('.')
    #     for handler in self.event_handlers:
    #         if hasattr(handler,event_str):
    #             getattr(handler,event_str)(**state)

    # def raise_flag(self, flag):
    #     self.dat['flag'] = flag

    # def train(self, model, dataset):
    #     self.raise_flag(flag=luz.Flag.TRAINING)
    #     loader = model.dataloader(dataset=dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=self.pin_memory)

    #     model.train()
    #     model.to(device=self.device)
    #     model.device = self.device # FIXME: this is a hacky way to give model a device so that it can migrate tensors to the same device - fix this

    #     # must come after model.to
    #     optimizer = self.optimizer.link_model(model=model)
    #     #self.cur_optimizer = optimizer

    #     self.dat.update({'model':model,'trainer':self,'dataset':dataset,'loader':loader,'optimizer':optimizer})
    #     self.call_event(luz.Event.TRAINING_STARTED,state=self.dat)

    #     for epoch in range(1,self.num_epochs+1): # FIXME: start at self.current_epoch rather than 1 to make restarting from checkpoint possible?
    #         self.dat.update({'epoch': epoch})
    #         self.call_event(luz.Event.EPOCH_STARTED,state=self.dat)

    #         first_batch = next(iter(loader)) # FIXME: debugging by overfitting first batch
    #         for index_,sample in enumerate(loader):#enumerate([first_batch]*1000):#enumerate(loader):
    #             self.dat.update({'index': index_,'sample': sample})
    #             self.call_event(event=luz.Event.BATCH_STARTED,state=self.dat)

    #             output,loss,data,target = model.train_sample(sample,criterion=self.loss) # FIXME: should train_sample return not just output and loss but also the data and target from the sample?
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             self.dat.update({'output':output,'loss':loss,'data':data,'target':target})
    #             self.call_event(event=luz.Event.BATCH_ENDED,state=self.dat)
    #         #self.run_epoch(model=model,optimizer=optimizer,loader=loader)
    #         self.call_event(luz.Event.EPOCH_ENDED,state=self.dat)

    #     self.call_event(luz.Event.TRAINING_ENDED,state=self.dat)

    # def test(self, model, dataset):
    #     self.raise_flag(flag=luz.Flag.TESTING)
    #     loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=self.pin_memory)
    #     optimizer = self.optimizer.link_model(model=model)

    #     # Set the model to eval
    #     model.eval()

    #     self.dat.update({'model':model,'trainer':self,'dataset':dataset,'loader':loader,'optimizer':optimizer})
    #     self.call_event(luz.Event.TESTING_STARTED,state=self.dat)

    #     # for epoch in range(1,self.num_epochs+1):
    #     #     state['epoch'] = epoch
    #         #self.call_event(luz.Event.EPOCH_STARTED,state=self.dat)
    #         #self.callback.on_epoch_start(model=model,trainer=self,optimizer=optimizer,loader=loader,epoch=epoch)
    #     test_loss = self.run_epoch(model=model,optimizer=optimizer,loader=loader)
    #     #FIXME: should run_epoch be renamed? should it be different between train and test? should epoch be passed at all?
    #         #self.callback.on_epoch_end(model=model,trainer=self,optimizer=optimizer,loader=loader,epoch=epoch)
    #         #self.call_event(luz.Event.EPOCH_ENDED,state=self.dat)

    #     self.call_event(luz.Event.TESTING_ENDED,state=self.dat)

    #     return test_loss

    # def run_epoch(self, model, optimizer, loader, epoch):
    #     raise NotImplementedError

    # def to(self, device):
    #     self.device = device



# """

# Trainer class for training machine learning models.

# """

# import torch.utils.data

# import luz

# class Trainer:
#     """
#     Reads in data from file, splits data into training and validation sets, constructs the loss function and optimizer, trains the model, and executes any callbacks.
#     ...

#     Attributes
#     ----------
#     optim_func: obj:`torch.nn.optimizer`
#         Optimizer function used during model training.

#     optim_args: obj:`dict`
#         Keyword arguments which will be passed to optim_func when the optimizer is constructed.

#     loss: obj:`torch.nn._Loss`
#         Loss function which measures the training and validation error of the model.

#     num_epochs: int
#         Maximum number of epochs of training.

#     train_loader: obj:`torch.utils.data.DataLoader`
#         DataLoader for training data.

#     callbacks: obj:`list` of obj:`luz_stable.Callback`
#         Callback functions to be executed before, during, and after training each epoch.
#     """

#     def __init__(self, num_epochs, batch_size, shuffle, loss, optimizer, num_workers=1, pin_memory=False, event_handlers=()):
#         self.device = None

#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         self.num_epochs = num_epochs

#         self.loss = loss

#         self.optimizer = optimizer

#         self.num_workers = num_workers
#         self.pin_memory = pin_memory

#         self.event_handlers = event_handlers

#         self.dat = {}

#     @classmethod
#     def builder(cls, *args, **kwargs):
#         def f(*new_args,**new_kwargs):
#             return cls(*args,*new_args,**kwargs,**new_kwargs)
#         return f

#     def state_dict(self):
#         return {'loss': self.loss, 'optimizer': self.cur_optimizer.state_dict(),
#                        'event_handlers': self.event_handlers,
#                        'num_epochs': self.num_epochs, 'batch_size': self.batch_size,
#                        'shuffle': self.shuffle, 'device': self.device}

#     def load_state_dict(self, state_dict):
#         self.loss = state_dict['loss']
#         self.cur_optimizer.load_state_dict(state_dict['optimizer'])
#         self.event_handlers = state_dict['event_handlers']
#         self.num_epochs = state_dict['num_epochs']
#         self.batch_size = state_dict['batch_size']
#         self.shuffle = state_dict['shuffle']
#         self.device = state_dict['device']

#     def call_event(self, event, state):
#         _,event_str = str(event).lower().rsplit('.') #Event.TRAINING_STARTED -> 'Event.TRAINING_STARTED' -> 'event.training_started' -> 'training_started'
#         for handler in self.event_handlers:
#             if hasattr(handler,event_str):
#                 getattr(handler,event_str)(**state)

#     def raise_flag(self, flag):
#         self.dat['flag'] = flag

#     def train(self, model, dataset):
#         self.raise_flag(flag=luz.Flag.TRAINING)
#         loader = model.dataloader(dataset=dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=self.pin_memory)

#         model.train()
#         model.to(device=self.device)
#         model.device = self.device # FIXME: this is a hacky way to give model a device so that it can migrate tensors to the same device - fix this

#         # must come after model.to
#         optimizer = self.optimizer.link_model(model=model)
#         #self.cur_optimizer = optimizer

#         self.dat.update({'model':model,'trainer':self,'dataset':dataset,'loader':loader,'optimizer':optimizer})
#         self.call_event(luz.Event.TRAINING_STARTED,state=self.dat)

#         for epoch in range(1,self.num_epochs+1): # FIXME: start at self.current_epoch rather than 1 to make restarting from checkpoint possible?
#             self.dat.update({'epoch': epoch})
#             self.call_event(luz.Event.EPOCH_STARTED,state=self.dat)

#             first_batch = next(iter(loader)) # FIXME: debugging by overfitting first batch
#             for index_,sample in enumerate(loader):#enumerate([first_batch]*1000):#enumerate(loader):
#                 self.dat.update({'index': index_,'sample': sample})
#                 self.call_event(event=luz.Event.BATCH_STARTED,state=self.dat)

#                 output,loss,data,target = model.train_sample(sample,criterion=self.loss) # FIXME: should train_sample return not just output and loss but also the data and target from the sample?
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 self.dat.update({'output':output,'loss':loss,'data':data,'target':target})
#                 self.call_event(event=luz.Event.BATCH_ENDED,state=self.dat)
#             #self.run_epoch(model=model,optimizer=optimizer,loader=loader)
#             self.call_event(luz.Event.EPOCH_ENDED,state=self.dat)

#         self.call_event(luz.Event.TRAINING_ENDED,state=self.dat)

#     def test(self, model, dataset):
#         self.raise_flag(flag=luz.Flag.TESTING)
#         loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=self.pin_memory)
#         optimizer = self.optimizer.link_model(model=model)

#         # Set the model to eval
#         model.eval()

#         self.dat.update({'model':model,'trainer':self,'dataset':dataset,'loader':loader,'optimizer':optimizer})
#         self.call_event(luz.Event.TESTING_STARTED,state=self.dat)

#         # for epoch in range(1,self.num_epochs+1):
#         #     state['epoch'] = epoch
#             #self.call_event(luz.Event.EPOCH_STARTED,state=self.dat)
#             #self.callback.on_epoch_start(model=model,trainer=self,optimizer=optimizer,loader=loader,epoch=epoch)
#         test_loss = self.run_epoch(model=model,optimizer=optimizer,loader=loader)
#         #FIXME: should run_epoch be renamed? should it be different between train and test? should epoch be passed at all?
#             #self.callback.on_epoch_end(model=model,trainer=self,optimizer=optimizer,loader=loader,epoch=epoch)
#             #self.call_event(luz.Event.EPOCH_ENDED,state=self.dat)

#         self.call_event(luz.Event.TESTING_ENDED,state=self.dat)

#         return test_loss

#     def run_epoch(self, model, optimizer, loader, epoch):
#         raise NotImplementedError

#     def to(self, device):
#         self.device = device

#     # !!! There's no current way to save loss or optim_func as basic data types - does Blaze need wrapper classes for these two that have state_dict() functions which can give basic dicts? Probably yes
#     # !!! Alternatively, all information which can be read from the config file will simply be reread, and the state_dict will be used to reconstruct specific state information
#     # !!! This latter possibility is more in line with how state_dict is already used in PyTorch

#     # def state(self):
#     #     return {'loss': self.loss, 'optimizer': self.optimizer,
#     #             'event_handlers': self.event_handlers,
#     #             'device': self.device,
#     #             #'callbacks': [cb.state_dict() for cb in self.callbacks],
#     #             #'metrics': [m.state_dict() for m in self.metrics],
#     #             'num_epochs': self.num_epochs, 'batch_size': self.batch_size,
#     #             'shuffle': self.shuffle, 'device': self.device}
#     #     # return {'optimizer': optimizer.state_dict(), 'loss_func': self.loss_func,
#     #     #         'optim_func': self.optim_func, 'optim_kwargs': self.optim_kwargs,
#     #     #         'callbacks': [cb.state_dict() for cb in self.callbacks],
#     #     #         'metrics': [m.state_dict() for m in self.metrics],
#     #     #         'num_epochs': self.num_epochs, 'batch_size': self.batch_size,
#     #     #         'shuffle': self.shuffle, 'device': self.device}
#     #
#     # # !!! Current training epoch should probably be saved/loaded as well
#     # def load_state_dict(self, state_dict):
#     #     self.device = state_dict['device']
#     #     self.batch_size = state_dict['batch_size']
#     #     self.shuffle = state_dict['shuffle']
#     #     self.num_epochs = state_dict['num_epochs']
#     #     self.loss = state_dict['loss']
#     #     self.optimizer = state_dict['optimizer']
#     #     #FIXME: load event handlers!
#     #     # for cb,cb_state_dict in zip(self.callbacks,state_dict['callbacks']):
#     #     #     cb.load_state_dict(cb_state_dict)
#     #     # for m,m_state_dict in zip(self.metrics,state_dict['metrics']):
#     #     #     m.load_state_dict(m_state_dict)

#         #self.optimizer.load_state_dict(state_dict['optimizer'])

#     # def load_data(self, dataset):
#     #     self.dataset = dataset
#     #
#     #     self.train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

# #     def train(self, model, dataset):
# #         # FIXME: how should num_workers be set? probably as a passable parameter for Trainer
# #         train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=4,pin_memory=True)
# #         # save_dir = os.path.join(os.path.split(sys.argv[0])[0],'models','{}_checkpoints').format(model.model_name)
# #         #
# #         # try:
# #         #     os.makedirs(save_dir)
# #         # except OSError as e:
# #         #     if e.errno != errno.EEXIST:
# #         #         raise
# #
# #         # Initialize the optimizer
# #         #optimizer = self.optim_func(model.parameters(), **self.optim_kwargs)
# #         optimizer = self.optimizer.link_model(model=model)
# #
# #         # # Initialize the callbacks
# #         # for cb in self.callback:
# #         #     cb.compile(model=model, trainer=self)#dataloader=self.train_loader)
# #
# #         # Set the model to train
# #         model.train()
# #
# #         for epoch in range(1,self.num_epochs+1):
# #             #self._call_callbacks(call_time='start')
# #             #self.pre_epoch()
# #             #self.epoch()
# #             #self.post_epoch()
# #             #self.callback.on_epoch_start()
# #             self.train_epoch(model=model,optimizer=optimizer,epoch_num=epoch,train_loader=train_loader)
# #
# # #            callback_returns = self.callback.on_epoch_end()#self._call_callbacks(call_time='end')
# #
# #             # if any([cbr == False for cbr in callback_returns]):
# #             #     break
# #
# #         print('Training complete.')
# #
# #         # Set the model to eval
# #         model.eval()
# #
# #     def pre_epoch(self):
# #         raise NotImplementedError
# #
# #     def epoch(self):
# #         raise NotImplementedError
# #
# #     def post_epoch(self):
# #         raise NotImplementedError

#     #def _train_epoch(self, model, optimizer, epoch_num, train_loader):
#         #raise NotImplementedError

#     # def _call_callbacks(self, call_time):
#     #     if call_time == 'start':
#     #         return [callback.on_epoch_start() for callback in self.callbacks]
#     #     if call_time == 'during':
#     #         return [callback.during_epoch() for callback in self.callbacks]
#     #     if call_time == 'end':
#     #         return [callback.on_epoch_end() for callback in self.callbacks]


#         #else:
#             #return None

#     # def train_val_split(self):
#     #     points_per_fold = math.ceil(len(self.trainval_dataset)//self.num_folds)
#     #     fold_lengths = [points_per_fold]*self.num_folds
#     #     fold_lengths[-1] -= sum(fold_lengths) - len(self.trainval_dataset)
#     #
#     #     folds = torch.utils.data.random_split(dataset=self.trainval_dataset,lengths=fold_lengths)
#     #
#     #     for i in range(self.num_folds):
#     #         train_dataset = torch.utils.data.ConcatDataset(datasets=[f for j,f in enumerate(folds) if j != i])
#     #         val_dataset = folds[i]
#     #
#     #         self.train_length = len(train_dataset)
#     #         self.val_length = len(val_dataset)
#     #
#     #         train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True)
#     #         val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True)
#     #
#     #         yield train_loader,val_loader
#     #
#     #     def validate(self, model):
#     #     print("Performing {0}-fold cross-validation.".format(self.num_folds))
#     #     total_val_loss = 0
#     #     for index,(train_loader,val_loader) in enumerate(self.train_val_split()):
#     #         print("Validating on fold {0}.".format(index+1))
#     #         self.train_loader = train_loader
#     #         self.val_loader = val_loader
#     #         self.train(model=model)
#     #         total_val_loss += self.evaluate(model=model)
#     #     return total_val_loss/self.num_folds
#     #
#     # @contextmanager
#     # def optional(condition, context_manager):
#     # if condition:
#     #     with context_manager:
#     #         yield
#     # else:
#     #     yield
#     #
#     # def evaluate(self, model, loader, no_grad):
#     #     with self.val_loader.dataset.dataset.dataset.eval():
#     #         with torch.no_grad():
#     #             running_loss = 0.0
#     #             criterion = self.loss_func()
#     #
#     #             for x,y in self.val_loader:
#     #                 x, y = x.to(self.device), y.to(self.device)
#     #
#     #                 output = model.forward(x)
#     #
#     #                 loss = criterion(output, y)
#     #
#     #                 running_loss += loss.item()
#     #
#     #     return running_loss/len(self.val_loader)
#     #
#     # def test(self, model):
#     #     if len(self.test_loader) > 0:
#     #         with self.test_loader.dataset.dataset.eval():
#     #
#     #             running_loss = 0.0
#     #             criterion = self.loss_func()
#     #
#     #             for x,y in self.test_loader:
#     #                 # Migrate the input and target tensors to the appropriate device
#     #                 x, y = x.to(self.device), y.to(self.device)
#     #
#     #                 output = model.forward(x=x)
#     #
#     #                 if self.metrics is not None:
#     #                     for metric in self.metrics:
#     #                         metric.update(x=x,y=y,predicted=output)
#     #
#     #                 loss = criterion(output, y)
#     #                 running_loss += loss.item()
#     #
#     #             if self.metrics is not None:
#     #                 for metric in self.metrics:
#     #                     metric.compute()
#     #
#     #             #print('Test loss: {0}'.format(running_loss/self.test_length))
#     #             print('Test loss: {0}'.format(running_loss/len(self.test_loader)))
#     #
#     #         return running_loss#/self.test_length
#     #     else:
#     #         return None
