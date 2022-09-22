from pytorch_lightning import LightningDataModule


class BaseLDM(LightningDataModule):
  def __init__(self, batch_size=2, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLDM")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--batch_size", type=int)
    return parent_parser