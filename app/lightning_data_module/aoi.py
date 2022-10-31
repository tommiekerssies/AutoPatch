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
from app.dataset.aoi import AOI as AOIDataset


class AOI(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--crop_size", type=int)
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--val_batch_size", type=int)

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

        self.train_dataset = AOIDataset(
            os.path.join(self.dataset_path, "train_non_ignore_buffer00"),
            transform=Compose(
                augmentations + preprocessing, additional_targets=additional_targets
            ),
        )
        self.val_dataset = AOIDataset(
            os.path.join(self.dataset_path, "val_cropped_non_ignore_buffer00"),
            transform=Compose(preprocessing, additional_targets=additional_targets),
        )
        self.predict_dataset = AOIDataset(
            os.path.join(self.dataset_path, "train_non_ignore_buffer00"),
            transform=Compose(preprocessing, additional_targets=additional_targets),
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
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
