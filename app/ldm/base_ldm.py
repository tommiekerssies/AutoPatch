from pytorch_lightning import LightningDataModule
from torch import Generator


class BaseLDM(LightningDataModule):
  def __init__(self, num_workers, batch_size, seed, **kwargs):
    super().__init__()
    self.dataloader_kwargs = dict(
      batch_size=batch_size, 
      num_workers=num_workers, pin_memory=True,
      generator=Generator().manual_seed(seed))
    
  @staticmethod
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("BaseLDM")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int)
    return parent_parser