import os
from typing import Optional
from torch.utils.data import DataLoader
from app.lightning_data_module.base import Base
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    CropNonEmptyMaskIfExists,
)
from app.dataset.aoi_multilabel_v6 import AOI as AOIDataset


class AOI(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--crop_size", type=int)
        parser.add_argument("--use_augmentations", action="store_true")

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        super().__init__()
        self.dataset_path = os.path.join(self.hparams.work_dir, "multilabel_v6")

    def setup(self, stage: Optional[str] = None):
        augmentations = []
        if self.hparams.crop_size:
            augmentations.append(
                CropNonEmptyMaskIfExists(
                    self.hparams.crop_size, self.hparams.crop_size
                ),
            )
        if self.hparams.use_augmentations:
            augmentations += [
                HorizontalFlip(),
                VerticalFlip(),
                Rotate(limit=(90, 90)),  # type: ignore
            ]

        preprocessing = [ToTensorV2()]

        additional_targets = {"ignore_mask": "mask"}

        self.train_dataset = AOIDataset(
            os.path.join(self.dataset_path, "train"),
            transform=Compose(
                augmentations + preprocessing, additional_targets=additional_targets
            ),
        )
        self.val_dataset = AOIDataset(
            os.path.join(self.dataset_path, "val"),
            transform=Compose(preprocessing, additional_targets=additional_targets),
        )
        self.predict_dataset = AOIDataset(
            os.path.join(self.dataset_path, "train"),
            transform=Compose(preprocessing, additional_targets=additional_targets),
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, **self.dataloader_kwargs)
