from typing import Any
from pytorch_lightning import LightningDataModule
from torch import Generator


class Base(LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--batch_size", type=int)

    def __init__(self, pin_memory=True, **kwargs):
        super().__init__()
        self.hparams: Any
        self.pin_memory = pin_memory

    @property
    def dataloader_kwargs(self):
        return dict(
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            generator=self.generator(),
            persistent_workers=self.hparams.num_workers > 0,
        )
        
    def generator(self):
        return Generator().manual_seed(self.hparams.seed)
