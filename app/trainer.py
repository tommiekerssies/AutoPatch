from argparse import Namespace
import inspect
from math import ceil
from pytorch_lightning import Trainer as TrainerPL
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only


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
    if not lr:
      lm.hparams.lr = 1e-3
    
    tune_trainer = TrainerPL.from_argparse_args(
      Namespace(**self.trainer_kwargs),
      strategy=None, devices=[self.global_rank], num_nodes=1, 
      auto_lr_find=False if lr else True,
      auto_scale_batch_size=None if batch_size else 'binsearch')

    tune_trainer.tune(lm, datamodule=datamodule)
    
    # Reduce batch size by 2% because the found batch size is too tight
    datamodule.hparams.batch_size = \
      ceil(datamodule.hparams.batch_size * 0.98)
    
    print(datamodule.hparams.batch_size)
    # Scale learning rate by number of participating GPUs,
    # based on linear scaling rule from http://arxiv.org/abs/1706.02677
    lr_multiplier = 1
    if devices:
      lr_multiplier *= int(devices)
    if num_nodes:
      lr_multiplier *= int(num_nodes)
    lm.hparams.lr *= lr_multiplier
        
  def add_argparse_args(*args, **kwargs):
    return TrainerPL.add_argparse_args(*args, **kwargs)