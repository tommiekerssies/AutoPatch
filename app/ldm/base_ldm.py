from pytorch_lightning import LightningDataModule
from torch import Generator


class BaseLDM(LightningDataModule):
  def __init__(self, **kwargs):
    super().__init__()
    
  @property
  def dataloader_kwargs(self):
    return dict(
      batch_size=self.hparams.batch_size, 
      num_workers=self.hparams.num_workers, pin_memory=True,
      generator=Generator().manual_seed(self.hparams.seed))
  
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLDM")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int)
    return parent_parser