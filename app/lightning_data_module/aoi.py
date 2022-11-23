import os
from random import Random
from typing import Optional
from torch.utils.data import DataLoader, Subset
from app.lightning_data_module.base import Base
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    CropNonEmptyMaskIfExists,
)
from app.dataset.aoi import AOI as AOIDataset


class AOI(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--crop_size", type=int)
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--val_batch_size", type=int)
        parser.add_argument("--train_fraction", type=float, default=1.0)
        parser.add_argument("--train_folder", type=str, default="train_buffer00_only_wire")
        parser.add_argument("--val_folder", type=str, default="val_cropped_buffer00_only_wire")

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)
        self.dataset_path = os.path.join(self.hparams.work_dir, "multilabel_v6")

    def setup(self, stage: Optional[str] = None):
        self.val_batch_size = self.hparams.val_batch_size or self.hparams.batch_size

        augmentations = []
        if self.hparams.crop_size:
            augmentations.append(
                CropNonEmptyMaskIfExists(
                    self.hparams.crop_size, self.hparams.crop_size
                ),
            )
        if self.hparams.augment:
            augmentations += [
                HorizontalFlip(),
                VerticalFlip(),
                Rotate(limit=(90, 90)),  # type: ignore
            ]

        preprocessing = [ToTensorV2()]

        additional_targets = {"ignore_mask": "mask"}

        self.train_path = os.path.join(self.dataset_path, self.hparams.train_folder)
        self.val_path = os.path.join(
            self.dataset_path, self.hparams.val_folder
        )

        self.train_dataset = AOIDataset(
            self.train_path,
            transform=Compose(
                augmentations + preprocessing, additional_targets=additional_targets
            ),
        )
        
        if self.hparams.train_fraction < 1.0:
            indices = list(range(len(self.train_dataset)))
            Random(self.hparams.seed).shuffle(indices)
            indices = indices[:int(len(self.train_dataset) * self.hparams.train_fraction)]
            self.train_dataset = Subset(self.train_dataset, indices)
        
        self.val_dataset = AOIDataset(
            self.val_path,
            transform=Compose(preprocessing, additional_targets=additional_targets),
        )
        self.predict_dataset = AOIDataset(
            self.val_path,
            transform=Compose(preprocessing, additional_targets=additional_targets),
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size or 1,
            shuffle=True,
            drop_last=True,
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, **self.dataloader_kwargs
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.val_batch_size,
            **self.dataloader_kwargs
        )
