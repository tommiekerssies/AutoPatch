from torch.utils.data import DataLoader
from app.lightning_data_module.base import Base
from torchvision.datasets import CIFAR100 as CIFAR100Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


class CIFAR100(Base):
    def __init__(self, **kwargs):
        self.save_hyperparameters()
        super().__init__()

    def prepare_data(self):
        CIFAR100Dataset(root=self.hparams.work_dir, train=True, download=True)
        CIFAR100Dataset(root=self.hparams.work_dir, train=False, download=True)

    def setup(self, stage):
        preprocessing = [
            Resize(128),
            ToTensor(),
        ]
        augmentations = [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ]

        self.train_dataset = CIFAR100Dataset(
            root=self.hparams.work_dir,
            transform=Compose(augmentations + preprocessing),
            train=True,
        )
        self.val_dataset = CIFAR100Dataset(
            root=self.hparams.work_dir, transform=Compose(preprocessing), train=False
        )
        self.predict_dataset = CIFAR100Dataset(
            root=self.hparams.work_dir, transform=Compose(preprocessing), train=True
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            **self.dataloader_kwargs
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            **self.dataloader_kwargs
        )
