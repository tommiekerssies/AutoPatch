from typing import Any
from pytorch_lightning import LightningDataModule
from torch import Generator
from warnings import filterwarnings


class Base(LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=267)
        parser.add_argument("--pin_memory", action="store_true")

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams: Any

    @property
    def dataloader_kwargs(self):
        return dict(
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            generator=self.generator(),
            persistent_workers=self.hparams.num_workers > 0,
            batch_size=self.hparams.batch_size,
        )

    def generator(self):
        return Generator().manual_seed(self.hparams.seed)


filterwarnings("ignore", ".*does not have many workers.*")
