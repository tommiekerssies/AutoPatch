from argparse import Namespace
import inspect
from pytorch_lightning import Trainer as TrainerPL
from pytorch_lightning.callbacks import ModelCheckpoint


class Trainer(TrainerPL):
  def __init__(self, monitor_=None, logger_=None,
               fast_dev_run_=False, max_epochs_=-1,
               **kwargs):
    callbacks = []
    if monitor_:
      callbacks.append(ModelCheckpoint(
        monitor=monitor_, mode='max', save_last=True,
        filename=f'{{epoch}}-{{{monitor_}:.4f}}'))
    
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(TrainerPL.__init__).parameters
    self.trainer_kwargs = {name: kwargs[name] for name
                      in valid_kwargs if name in kwargs}
    
    self.trainer_kwargs.update(dict(
      strategy='ddp_find_unused_parameters_false',
      accelerator='auto', log_every_n_steps=1,
      deterministic=True, callbacks=callbacks,
      max_epochs=max_epochs_ or kwargs['max_epochs'],
      logger=logger_ or kwargs['logger'], 
      fast_dev_run=fast_dev_run_ or kwargs['fast_dev_run']))

    super().__init__(**self.trainer_kwargs)
    
  def tune(self, lm, datamodule, devices, num_nodes,
           lr, batch_size, **kwargs):    
    tune_trainer = TrainerPL.from_argparse_args(
      Namespace(**self.trainer_kwargs),
      strategy=None, devices=1, num_nodes=1, 
      auto_lr_find=True,
      auto_scale_batch_size='binsearch')

    tune_trainer.tune(lm, datamodule=datamodule)
        
  def add_argparse_args(*args, **kwargs):
    return TrainerPL.add_argparse_args(*args, **kwargs)