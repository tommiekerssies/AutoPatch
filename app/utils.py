from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger
import wandb
from os.path import join
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
from pytorch_lightning import Trainer as TrainerPL


def add_global_args(parent_parser):
  parser = parent_parser.add_argument_group("Global")
  parser.add_argument("--seed", type=int)
  parser.add_argument("--resume_run_id", type=str)
  parser.add_argument("--work_dir", type=str)
  parser.add_argument("--weights_file", type=str)
  parser.add_argument("--prefix_old", type=str)
  parser.add_argument("--prefix_new", type=str)
  parser.add_argument("--project_name", type=str)
  return parent_parser

def get_kwargs(*classes):
  parser = ArgumentParser()
  parser = add_global_args(parser)
  for cls in classes:
    parser = cls.add_argparse_args(parser)
  return vars(parser.parse_args())

def get_lm(cls, resume_run_id, work_dir, project_name,
           **kwargs):
  ckpt_path = None
  if resume_run_id:
    ckpt_path = join(work_dir, project_name,
                     resume_run_id, "checkpoints", 
                     "last.ckpt")
    lm = cls.load_from_checkpoint(ckpt_path, **kwargs)
  else:
    lm = cls(**kwargs)
  return lm, ckpt_path

def load_weights(lm, resume_run_id, weights_file, work_dir,
                 prefix_old, prefix_new, **kwargs):
  if resume_run_id or not weights_file:
    return lm
  
  obj = torch.load(join(work_dir, weights_file))
  return _load_state_dict_from_obj(lm, obj, prefix_old, 
                                   prefix_new)

def load_subnet_weights(supernet_cls, lm, ldm, resume_run_id,
                        weights_file, work_dir, prefix_old,
                        prefix_new, **kwargs):
  if resume_run_id or not weights_file:
    return lm
  
  obj = torch.load(join(work_dir, weights_file))
  supernet_lm = _load_state_dict_from_obj(supernet_cls(), obj,
                                          prefix_old, 
                                          prefix_new)
  
  # Doing a forward pass with supernet in deploy mode should
  # remove unused layers and channels from its state dict.
  supernet_lm.model.deploy()
  supernet_lm.model.manipulate_arch(dict(
      encoder_q=dict(
        stem=dict(width=lm.model_cfg['backbone']['stem_width']),
        body=dict(width=lm.model_cfg['backbone']['body_width'],
                  depth=lm.model_cfg['backbone']['body_depth']))))
  TrainerPL.from_argparse_args(Namespace(**kwargs), fast_dev_run=True) \
    .predict(supernet_lm, datamodule=ldm)
    
  return _load_state_dict_from_obj(lm, supernet_lm)

def _load_state_dict_from_obj(lm, obj, prefix_old=None, 
                              prefix_new=None):
  if getattr(obj, 'state_dict', None):
    state_dict = obj.state_dict()
  
  elif isinstance(obj, dict):
    if 'state_dict' in obj:
      state_dict = obj['state_dict']
    else:
      state_dict = obj
  
  else:
    raise ValueError("Object must contain a state dict.")
  
  new_state_dict = {}
  for key, value in state_dict.items():
    prefix_old = prefix_old or ""
    prefix_new = prefix_new or ""
    new_key = prefix_new + key.replace(prefix_old, "")
    new_state_dict[new_key] = value
  
  missing_keys, unexpected_keys = lm.load_state_dict(
    new_state_dict, strict=False)
  
  imported_keys = new_state_dict.keys() - unexpected_keys
  
  rank_zero_info(f"Imported keys: {imported_keys}")
  rank_zero_info(f"Missing keys: {missing_keys}")
  rank_zero_info(f"Unexpected keys: {unexpected_keys}")
  
  if len(imported_keys) == 0:
    raise ValueError("No keys were imported.")
  
  return lm
  

def get_logger(lm, resume_run_id, project_name, 
               work_dir, **kwargs):
  logger = WandbLogger(id=resume_run_id, project=project_name,
                       save_dir=work_dir,
                       resume='must' if resume_run_id else 'never',
                       settings=wandb.Settings(code_dir="."))
  logger.watch(lm)
  return logger