from typing import Any
from pytorch_lightning import LightningDataModule
from torch import Generator


class Base(LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int)

    def __init__(self):
        super().__init__()
        self.hparams: Any

    @property
    def dataloader_kwargs(self):
        return dict(
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            generator=Generator().manual_seed(self.hparams.seed),
        )
