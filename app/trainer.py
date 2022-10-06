from argparse import Namespace
import inspect
from pytorch_lightning import Trainer as TrainerPL
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class Trainer(TrainerPL):
  def __init__(self, patience, monitor_=None, logger=None, 
               fast_dev_run=None, max_epochs=None, **kwargs):
    callbacks = []
    if monitor_:
      callbacks.append(ModelCheckpoint(
        monitor=monitor_, mode='max', save_last=True,
        filename=f'{{epoch}}-{{{monitor_}:.4f}}'))
      callbacks.append(EarlyStopping(
        monitor=monitor_, mode='max', patience=patience + 1))
  
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(TrainerPL.__init__).parameters
    self.trainer_kwargs = {name: kwargs[name] for name
                      in valid_kwargs if name in kwargs}

    self.trainer_kwargs.update(dict(
      strategy='ddp_find_unused_parameters_false',
      accelerator='auto', log_every_n_steps=1,
      deterministic=True, callbacks=callbacks,
      max_epochs=max_epochs or -1,
      logger=logger,
      fast_dev_run=fast_dev_run))

    super().__init__(**self.trainer_kwargs)
    
  def tune(self, lm, datamodule, lr, batch_size, **kwargs):    
    auto_lr_find = not lr
    auto_scale_batch_size = not batch_size
    
    if auto_lr_find:
      lm.hparams.lr = 1e-3
    lm.hparams.warmup_steps = 0
    
    tune_trainer = TrainerPL.from_argparse_args(
      Namespace(**self.trainer_kwargs),
      strategy=None, devices=1, num_nodes=1, 
      auto_lr_find=auto_lr_find,
      auto_scale_batch_size=auto_scale_batch_size)

    tune_trainer.tune(lm, datamodule=datamodule)
        
  def add_argparse_args(*args, **kwargs):
    return TrainerPL.add_argparse_args(*args, **kwargs)