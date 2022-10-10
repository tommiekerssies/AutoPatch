from torch.utils.data import DataLoader
from app.ldm.base_ldm import BaseLDM
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, RandomCrop,\
  RandomHorizontalFlip, Resize, ToTensor

class CIFAR100LDM(BaseLDM):
  def __init__(self, **kwargs):
    super().__init__()
    self.save_hyperparameters()

  def prepare_data(self):
    CIFAR100(root=self.hparams.work_dir, train=True, download=True)
    CIFAR100(root=self.hparams.work_dir, train=False, download=True)

  def setup(self, stage):
    preprocessing = [
      Resize(128),
      ToTensor(),
    ]
    augmentations = [
      RandomCrop(32, padding=4),
      RandomHorizontalFlip(),
    ]
    
    self.train = CIFAR100(root=self.hparams.work_dir,
                          transform=Compose(augmentations + preprocessing), 
                          train=True)
    self.val = CIFAR100(root=self.hparams.work_dir, 
                        transform=Compose(preprocessing),
                        train=False)
    return self

  def train_dataloader(self):
    return DataLoader(self.train, shuffle=True, drop_last=False,
                      **self.dataloader_kwargs)

  def val_dataloader(self):
    return DataLoader(self.val, shuffle=False, drop_last=False,
                      **self.dataloader_kwargs)
    
  def predict_dataloader(self):
    return DataLoader(self.train, shuffle=False, drop_last=False,
                      **self.dataloader_kwargs)