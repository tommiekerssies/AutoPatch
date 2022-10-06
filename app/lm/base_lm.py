from pytorch_lightning import LightningModule
from mmselfsup.models import build_algorithm


class BaseLM(LightningModule):
  def __init__(self, model_cfg, **kwargs):
    super().__init__()
    self.model = build_algorithm(model_cfg)
    self.log_args = dict(sync_dist=True, on_step=False,
                         on_epoch=True)
  
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