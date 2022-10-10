from pytorch_lightning import LightningModule
from mmselfsup.models import build_algorithm
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
from os.path import join


class BaseLM(LightningModule):
  @classmethod
  def create(cls, ldm, resume_run_id, work_dir, project_name,
           weights_file, **kwargs):
    ckpt_path = None
    
    if resume_run_id:
      ckpt_path = join(work_dir, project_name,
                      resume_run_id, "checkpoints", 
                      "last.ckpt")
      lm = cls.load_from_checkpoint(ckpt_path, **kwargs)
    
    else:      
      lm = cls(work_dir=work_dir, **kwargs)
      
      if weights_file:
        obj = torch.load(join(work_dir, weights_file),
                         map_location=lm.device)
        lm.load_weights(obj, ldm, **kwargs)
      
    lm.ckpt_path = ckpt_path
  
    return lm
  
  def __init__(self, model_cfg):
    super().__init__()
    
    self.log_args = dict(sync_dist=True, on_step=False,
                         on_epoch=True)
    
    self.model = build_algorithm(model_cfg)
      
  def load_weights(self, obj, ldm, prefix_old, prefix_new, **kwargs):
    return self.load_state_dict_from_obj(obj, prefix_old, 
                                         prefix_new)

  def load_state_dict_from_obj(self, obj, prefix_old=None, 
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
    
    missing_keys, unexpected_keys = self.load_state_dict(
      new_state_dict, strict=False)
    
    imported_keys = new_state_dict.keys() - unexpected_keys
    
    rank_zero_info(f"Imported keys: {imported_keys}")
    rank_zero_info(f"Missing keys: {missing_keys}")
    rank_zero_info(f"Unexpected keys: {unexpected_keys}")
    
    if len(imported_keys) == 0:
      raise ValueError("No keys were imported.")
    
    return self

  def optimizer_step(
      self,
      epoch,
      batch_idx,
      optimizer,
      optimizer_idx,
      optimizer_closure,
      on_tpu=False,
      using_native_amp=False,
      using_lbfgs=False,
  ):
    optimizer.step(closure=optimizer_closure)
    
    # learning rate warm-up
    if self.trainer.global_step < self.hparams.warmup_steps:
      lr_scale = float(self.trainer.global_step + 1) \
                 / self.hparams.warmup_steps
      for i, pg in enumerate(optimizer.param_groups):
        pg['lr'] = lr_scale * self.hparams.lr
        print(f"Step {self.trainer.global_step}: increasing learning rate"\
              f" of group {i} to {pg['lr']}.")
  
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLM")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_steps", type=int, default=0)
    return parent_parser