from typing import Optional
from torch.utils.data import DataLoader
from app.lightning_data_module.base import Base
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    Resize,
    Normalize
)
from app.dataset.mvtec import MVTec as MVTecDataset


class MVTec(Base):
    @staticmethod
    def add_argparse_args(parser):
        Base.add_argparse_args(parser)
        parser.add_argument("--category", type=str)
        
    def __init__(self, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None):
        transform = Compose([
            Resize(224, 224), 
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        self.dataset = MVTecDataset(self.hparams.category, self.hparams.work_dir, transform)
        return self

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            **self.dataloader_kwargs
        )
