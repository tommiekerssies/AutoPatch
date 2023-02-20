from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.transforms.functional import resize
from os import listdir
from torch import device, stack, zeros
from torch.utils.data import Dataset
from pathlib import Path
from PIL.Image import open


class MVTecDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        class_: str,
        max_img_size: int,
        test: bool,
    ):
        class_path = Path(dataset_dir, class_, "test" if test else "train")
        img_mask_pairs = []
        for subclass in listdir(class_path):
            subclass_path = Path(class_path, subclass)
            img_paths = [
                str(Path(subclass_path, img)) for img in sorted(listdir(subclass_path))
            ]
            if subclass != "good":
                anomaly_mask_path = Path(dataset_dir, class_, "ground_truth", subclass)
                mask_paths = [
                    str(Path(anomaly_mask_path, mask))
                    for mask in sorted(listdir(anomaly_mask_path))
                ]
            img_mask_pairs.extend(
                (img_path, mask_paths[i] if subclass != "good" else None)
                for i, img_path in enumerate(img_paths)
            )

        transforms = [
            Resize(max_img_size),
            ToTensor(),
        ]
        transform_img = Compose(
            transforms
            + [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        transform_mask = Compose(transforms)

        imgs = []
        masks = []
        for img_path, mask_path in img_mask_pairs:
            imgs.append(transform_img(open(img_path).convert("RGB")))
            masks.append(
                transform_mask(open(mask_path)).int()
                if mask_path is not None
                else zeros([1, *imgs[-1].size()[1:]], dtype=int)
            )

        self.imgs = stack(imgs)
        self.masks = stack(masks)
        self.img_size = max_img_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i: int):
        img, mask = self.imgs[i], self.masks[i]
        if self.img_size != img.shape[-1]:
            img = resize(img, self.img_size)
            mask = resize(mask, self.img_size)
        return img, mask

    def to(self, device: device):
        self.imgs = self.imgs.to(device)
        self.masks = self.masks.to(device)


class MVTecDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        class_: str,
        max_img_size: str,
        batch_size: int,
        test_batch_size: int,
        seed: int,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.class_ = class_
        self.max_img_size = max_img_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = MVTecDataset(
                self.dataset_dir,
                self.class_,
                self.max_img_size,
                test=False,
            )

        if not hasattr(self, "test_dataset"):
            self.test_dataset = MVTecDataset(
                self.dataset_dir,
                self.class_,
                self.max_img_size,
                test=True,
            )

        return self

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size
            if self.test_batch_size is not None
            else self.batch_size,
        )

    def to(self, device: device):
        self.train_dataset.to(device)
        self.test_dataset.to(device)

    def set_img_size(self, img_size: int):
        self.train_dataset.img_size = img_size
        self.test_dataset.img_size = img_size
