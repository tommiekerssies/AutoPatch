from typing import Optional
from torch.utils.data import DataLoader
from app.lightning_data_module.base import Base
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop, ToTensor
from app.dataset.mvtec import MVTec as MVTecDataset


class MVTec(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--resize", type=int, default=256)
        parser.add_argument("--img_size", type=int, default=224)
        parser.add_argument("--dataset_dir", type=str, default="MVTec")
        parser.add_argument("--category", type=str, default="temp")
        parser.add_argument("--train_holdout_ratio", type=float, default=0.5)
        parser.add_argument("--test_holdout_ratio", type=float, default=0.5)

    def __init__(self, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)

        transforms = [
            Resize(self.hparams.resize),
            CenterCrop(self.hparams.img_size),
            ToTensor(),
        ]
        self.transform_mask = Compose(transforms)
        self.transform_img = Compose(
            transforms
            + [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MVTecDataset(
            "train",
            self.hparams.work_dir,
            self.hparams.dataset_dir,
            self.hparams.category,
            self.hparams.train_holdout_ratio,
            self.transform_img,
        )
        self.train_holdout_dataset = MVTecDataset(
            "train",
            self.hparams.work_dir,
            self.hparams.dataset_dir,
            self.hparams.category,
            self.hparams.train_holdout_ratio,
            self.transform_img,
            holdout=True,
        )
        self.test_dataset = MVTecDataset(
            "test",
            self.hparams.work_dir,
            self.hparams.dataset_dir,
            self.hparams.category,
            self.hparams.test_holdout_ratio,
            self.transform_img,
            self.transform_mask,
        )
        self.test_holdout_dataset = MVTecDataset(
            "test",
            self.hparams.work_dir,
            self.hparams.dataset_dir,
            self.hparams.category,
            self.hparams.test_holdout_ratio,
            self.transform_img,
            self.transform_mask,
            holdout=True,
        )

        return self

    def predict_dataloader(self):
        return [
            DataLoader(self.train_dataset, **self.dataloader_kwargs),
            DataLoader(self.train_holdout_dataset, **self.dataloader_kwargs),
            DataLoader(self.test_dataset, **self.dataloader_kwargs),
            DataLoader(self.test_holdout_dataset, **self.dataloader_kwargs),
        ]
