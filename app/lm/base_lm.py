from pytorch_lightning import LightningModule
from mmselfsup.models import build_algorithm


class BaseLM(LightningModule):
  def __init__(self, model_cfg, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.model = build_algorithm(model_cfg)
    self.log_args = dict(sync_dist=True, on_step=False,
                         on_epoch=True)
    
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLM")
    parser.add_argument("--lr", type=float)
    return parent_parser