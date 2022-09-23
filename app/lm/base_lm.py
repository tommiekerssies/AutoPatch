from pytorch_lightning import LightningModule


class BaseLM(LightningModule):
  def __init__(self, lr, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    
    self.log_args = dict(sync_dist=True, on_step=False,
                         on_epoch=True)
    
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLM")
    parser.add_argument("--lr", type=float)
    return parent_parser