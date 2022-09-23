from pytorch_lightning import LightningDataModule


class BaseLDM(LightningDataModule):
  def __init__(self, num_workers, batch_size, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLDM")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int)
    return parent_parser