from typing import Optional
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.transforms.functional import resize
from pytorch_lightning import LightningDataModule
from os import listdir
from torch import device, stack, zeros
from torch.utils.data import Dataset
from pathlib import Path
from PIL.Image import open
from math import ceil


class MVTecDataset(Dataset):
    def __init__(
        self,
        split: str,
        work_dir: str,
        dataset_dir: str,
        max_img_size: int,
        val_ratio: float = None,
    ):
        dataset_path = Path(work_dir, dataset_dir)

        if split == "train":
            split_path = Path(dataset_path, "train")
        elif split in {"val", "test"}:
            split_path = Path(dataset_path, "test")

        subcategories = listdir(split_path)

        data_list = []
        for subcategory in subcategories:
            subcategory_path = Path(split_path, subcategory)
            img_paths = [
                str(Path(subcategory_path, img))
                for img in sorted(listdir(subcategory_path))
            ]

            if subcategory != "good":
                anomaly_mask_path = Path(dataset_path, "ground_truth", subcategory)
                mask_paths = [
                    str(Path(anomaly_mask_path, mask))
                    for mask in sorted(listdir(anomaly_mask_path))
                ]

            data_tuples = []
            for i, img_path in enumerate(img_paths):
                data_tuple = [img_path, None]
                if subcategory != "good":
                    data_tuple[1] = mask_paths[i]
                data_tuples.append(data_tuple)

            # If not train set, split into val and test set (as MVTec doesn't have a val set by default)
            if split != "train":
                i_first_test = ceil(0.5 * len(data_tuples))
                if split == "test":
                    # Make sure that test set is always the same set regardless of val_ratio
                    data_tuples = data_tuples[i_first_test:]
                elif subcategory == "good":
                    # Only use val_ratio for anomalies
                    data_tuples = data_tuples[:i_first_test]
                else:
                    data_tuples = data_tuples[: ceil(val_ratio * i_first_test)]

            data_list.extend(data_tuples)

        transforms = [
            Resize(max_img_size),
            ToTensor(),
        ]
        transform_mask = Compose(transforms)
        transform_img = Compose(
            transforms
            + [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        imgs = []
        masks = []
        for img_path, mask_path in data_list:
            imgs.append(transform_img(open(img_path).convert("RGB")))

            if mask_path is not None:
                mask = transform_mask(open(mask_path))
            else:
                mask = zeros([1, *imgs[-1].size()[1:]])

            masks.append(mask.int())

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
        work_dir: str,
        dataset_dir: str,
        max_img_size: int,
        batch_size: int,
        val_ratio: float,
    ):
        super().__init__()
        self.work_dir = work_dir
        self.dataset_dir = dataset_dir
        self.max_img_size = max_img_size
        self.batch_size = batch_size
        self.val_ratio = val_ratio

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = MVTecDataset(
                "train",
                self.work_dir,
                self.dataset_dir,
                self.max_img_size,
            )
        if not hasattr(self, "val_dataset"):
            self.val_dataset = MVTecDataset(
                "val",
                self.work_dir,
                self.dataset_dir,
                self.max_img_size,
                self.val_ratio,
            )
        if not hasattr(self, "test_dataset"):
            self.test_dataset = MVTecDataset(
                "test",
                self.work_dir,
                self.dataset_dir,
                self.max_img_size,
                self.val_ratio,
            )

        return self

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
