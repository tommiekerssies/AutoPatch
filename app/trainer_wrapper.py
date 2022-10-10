from argparse import Namespace
import inspect
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback,\
  ModelCheckpoint, EarlyStopping
import wandb
from lib.gaia.base_rule import SAMPLE_RULES
import lib.gaia.eval_rule as eval_rule
from mmcv.utils import build_from_cfg
from lib.gaia.model_space_manager import ModelSpaceManager
from os.path import join
from pytorch_lightning.loggers import WandbLogger


class ScheduledStopCallback(Callback):
  def __init__(self, stop_time):
    self.stop_time = None
    if stop_time:
      self.stop_time = datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

  def on_train_batch_end(self, trainer, *args, **kwargs):
    self.stop_if_time_reached(trainer)
 
  def on_test_batch_end(self, trainer, *args, **kwargs):
    self.stop_if_time_reached(trainer)
    
  def on_validation_batch_end(self, trainer, *args, **kwargs):
    self.stop_if_time_reached(trainer)
    
  def on_predict_batch_end(self, trainer, *args, **kwargs):
    self.stop_if_time_reached(trainer)    
  
  def stop_if_time_reached(self, trainer):
    if self.stop_time and datetime.now() >= self.stop_time:
      print('Scheduled stop time reached.')
      trainer.should_stop = True


class TrainerWrapper(Trainer):
  def __init__(self, seed, resume_run_id,
               project_name, work_dir, patience, stop_time,
               max_epochs, **kwargs):
    seed_everything(seed, workers=True)
    
    callbacks = [
      ModelCheckpoint(
        monitor='val_loss', mode='min', save_last=True,
        filename=f'{{epoch}}-{{val_loss:.4f}}', verbose=True),
      EarlyStopping(
        monitor='val_loss', mode='min', patience=patience + 2,
        verbose=True),
      ScheduledStopCallback(stop_time)]
  
    # we only want to pass in valid Trainer args, 
    # the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    self.trainer_kwargs = {name: kwargs[name] for name
                           in valid_kwargs if name in kwargs}

    logger = WandbLogger(id=resume_run_id, project=project_name,
                         save_dir=work_dir,
                         resume='must' if resume_run_id else 'never',
                         settings=wandb.Settings(code_dir="."))

    self.trainer_kwargs.update(dict(
      strategy='ddp_find_unused_parameters_false',
      accelerator='auto', log_every_n_steps=1,
      deterministic=True, callbacks=callbacks,
      max_epochs=max_epochs or -1,
      logger=logger))

    super().__init__(**self.trainer_kwargs)
    
  def fit(self, lm, ldm):
    self.logger.watch(lm)
    super().fit(lm, datamodule=ldm, ckpt_path=lm.ckpt_path)
    
  def search(self, lm, ldm, work_dir, model_space_file,
             min_gflops, max_gflops, **kwargs):
    model_space = ModelSpaceManager.load(join(work_dir, model_space_file))
    model_sampling_rule = build_from_cfg(dict(type='eval', func_str=
      f'lambda x: x[\'overhead.flops\'] >= {min_gflops * 1e9} ' +
      f'and x[\'overhead.flops\'] < {max_gflops * 1e9}'
    ), SAMPLE_RULES)

    subnet_candidates = model_space.ms_manager.apply_rule(model_sampling_rule)
    subnet_candidates = subnet_candidates.sample(frac=1).ms_manager.pack()

    best_distance = None
    for subnet in subnet_candidates:
      if 'encoder_k' in subnet['arch']:
        subnet['arch'].pop('encoder_k')
      lm.model.manipulate_arch(subnet['arch'])
      self.predict(lm, ldm)
      distance = lm.distance.compute()
      
      self.logger.experiment.log(dict(subnet=subnet, distance=distance))
      if best_distance is None or distance < best_distance:
        self.logger.experiment.log(dict(best_subnet=subnet, best_distance=distance))
        best_distance = distance
    
  def tune(self, lm, ldm, lr, batch_size, **kwargs):    
    auto_lr_find = not lr
    auto_scale_batch_size = 'binsearch' if not batch_size\
                                        else None

    # Set lr and bs to default values to prevent errors
    if auto_lr_find:
      lm.hparams.lr = 1e-3
    if auto_scale_batch_size:
      ldm.hparams.batch_size = 2
      
    # Don't perform lr warm-up
    lm.hparams.warmup_steps = 0
    
    tune_trainer = Trainer.from_argparse_args(
      Namespace(**self.trainer_kwargs),
      strategy=None, devices=1, num_nodes=1, 
      auto_lr_find=auto_lr_find,
      auto_scale_batch_size=auto_scale_batch_size)

    tune_trainer.tune(lm, datamodule=ldm)
        
  def add_argparse_args(parent_parser):
    parent_parser = Trainer.add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("Trainer")
    parser.add_argument("--stop_time", type=str)
    parser.add_argument("--model_space_file", type=str)
    parser.add_argument("--min_gflops", type=float)
    parser.add_argument("--max_gflops", type=float)
    return parent_parser
