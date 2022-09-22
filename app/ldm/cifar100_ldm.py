from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import Generator
from app.ldm.base_ldm import BaseLDM


class CIFAR100LDM(BaseLDM):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mean = [0.4914, 0.4822, 0.4465]
    self.std = [0.2023, 0.1994, 0.2010]

  def prepare_data(self):
    datasets.CIFAR100(root=self.hparams.work_dir, train=True, download=True)
    datasets.CIFAR100(root=self.hparams.work_dir, train=False, download=True)

  def setup(self, stage):
    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(self.mean, self.std)])
    self.train = datasets.CIFAR100(root=self.hparams.work_dir,
                                   transform=train_transform, 
                                   train=True)
    
    val_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(self.mean, self.std)])
    self.val = datasets.CIFAR100(root=self.hparams.work_dir, 
                                 transform=val_transform,
                                 train=False)

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.hparams.batch_size, 
                      num_workers=self.hparams.num_workers, pin_memory=True,
                      generator=Generator().manual_seed(self.hparams.seed),
                      shuffle=True, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.hparams.batch_size,
                      num_workers=self.hparams.num_workers, pin_memory=True,
                      generator=Generator().manual_seed(self.hparams.seed),
                      shuffle=False)
    
  def predict_dataloader(self):
    return DataLoader(self.train, batch_size=self.hparams.batch_size, 
                      num_workers=self.hparams.num_workers, pin_memory=True,
                      generator=Generator().manual_seed(self.hparams.seed),
                      shuffle=False)