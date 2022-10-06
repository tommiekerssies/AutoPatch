from torch.utils.data import DataLoader
from app.ldm.base_ldm import BaseLDM
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, RandomCrop,\
  RandomHorizontalFlip, Resize, ToTensor, Normalize

class CIFAR100LDM(BaseLDM):
  def __init__(self, num_workers, batch_size, seed, **kwargs):
    super().__init__(num_workers, batch_size, seed)
    self.save_hyperparameters()
    self.mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    self.std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

  def prepare_data(self):
    CIFAR100(root=self.hparams.work_dir, train=True, download=True)
    CIFAR100(root=self.hparams.work_dir, train=False, download=True)

  def setup(self, stage):
    preprocessing = [
      Resize(224),
      ToTensor(),
      Normalize(self.mean, self.std),
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