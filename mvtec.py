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
        category: str,
        max_img_size: int,
        k: int,
        split: str,
    ):
        num_holdout = len(listdir(Path(dataset_dir, category, "test", "good")))
        train_good_path = Path(dataset_dir, category, "train", "good")
        test_path = Path(dataset_dir, category, "test")
        mask_path = Path(dataset_dir, category, "ground_truth")

        if split == "train":
            train_good_files = sorted(listdir(train_good_path)[num_holdout:])
            img_paths = [Path(train_good_path, img) for img in train_good_files]
            img_mask_pairs = [(str(img_path), None) for img_path in img_paths]

        elif split == "val":
            val_good_files = sorted(listdir(train_good_path)[:num_holdout])
            img_paths = [Path(train_good_path, img) for img in val_good_files]
            img_mask_pairs = [(str(img_path), None) for img_path in img_paths]
            for subclass in listdir(test_path):
                if subclass == "good":
                    continue
                subclass_img_files = sorted(listdir(Path(test_path, subclass)))
                subclass_mask_files = sorted(listdir(Path(mask_path, subclass)))
                img_paths = [
                    str(Path(test_path, subclass, img)) for img in subclass_img_files
                ]
                mask_paths = [
                    str(Path(mask_path, subclass, mask)) for mask in subclass_mask_files
                ]
                img_mask_pairs.extend(list(zip(img_paths, mask_paths))[:k])

        elif split == "test":
            img_mask_pairs = []
            for subclass in sorted(listdir(test_path)):
                subclass_img_files = sorted(listdir(Path(test_path, subclass)))
                img_paths = [
                    Path(test_path, subclass, img) for img in subclass_img_files
                ]
                if subclass == "good":
                    img_mask_pairs.extend(
                        (str(img_path), None) for img_path in img_paths
                    )
                else:
                    subclass_mask_files = sorted(listdir(Path(mask_path, subclass)))
                    mask_paths = [
                        str(Path(mask_path, subclass, mask))
                        for mask in subclass_mask_files
                    ]
                    img_mask_pairs.extend(list(zip(img_paths, mask_paths))[k:])

        transforms = [
            ToTensor(),
            Resize(max_img_size, antialias=True),
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
                transform_mask(open(mask_path))
                if mask_path is not None
                else zeros([1, *imgs[-1].size()[1:]])
            )

        self.imgs = stack(imgs)
        self.masks = stack(masks)
        self.img_size = max_img_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i: int):
        img, mask = self.imgs[i], self.masks[i]
        if self.img_size != img.shape[-1]:
            img = resize(img, self.img_size, antialias=True)
            mask = resize(mask, self.img_size, antialias=True)
        # As image was resized there may be values between 0 and 1
        return img, (mask > 0).int().squeeze()

    def to(self, device: device):
        self.imgs = self.imgs.to(device)
        self.masks = self.masks.to(device)


class MVTecDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        category: str,
        max_img_size: str,
        batch_size: int,
        k: int,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.category = category
        self.max_img_size = max_img_size
        self.batch_size = batch_size
        self.k = k

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = MVTecDataset(
                self.dataset_dir,
                self.category,
                self.max_img_size,
                self.k,
                split="train",
            )

        if not hasattr(self, "val_dataset"):
            self.val_dataset = MVTecDataset(
                self.dataset_dir,
                self.category,
                self.max_img_size,
                self.k,
                split="val",
            )

        if not hasattr(self, "test_dataset"):
            self.test_dataset = MVTecDataset(
                self.dataset_dir,
                self.category,
                self.max_img_size,
                self.k,
                split="test",
            )

        return self

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def to(self, device: device):
        self.train_dataset.to(device)
        self.val_dataset.to(device)
        self.test_dataset.to(device)

    def set_img_size(self, img_size: int):
        self.train_dataset.img_size = img_size
        self.val_dataset.img_size = img_size
        self.test_dataset.img_size = img_size
