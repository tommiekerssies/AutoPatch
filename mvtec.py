from typing import Optional
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from pytorch_lightning import LightningDataModule
from os import listdir
from torch import stack, zeros
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pathlib import Path
from PIL.Image import open
from math import ceil


class MVTecDataset(Dataset):
    def __init__(
        self,
        split: str,
        dataset_path: str,
        transform_img: Compose,
        transform_mask: Compose = None,
        val_ratio: float = None,
    ):
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        dataset_path = Path(dataset_path)

        if split == "train":
            split_path = Path(dataset_path, "train")
        elif split in {"val", "test"}:
            split_path = Path(dataset_path, "test")

        subcategories = listdir(split_path)

        img_paths = {}
        mask_paths = {}

        self.data_list = []

        for subcategory in subcategories:
            subcategory_path = Path(split_path, subcategory)
            img_paths[subcategory] = [
                str(Path(subcategory_path, img))
                for img in sorted(listdir(subcategory_path))
            ]

            if subcategory != "good":
                anomaly_mask_path = Path(dataset_path, "ground_truth", subcategory)
                mask_paths[subcategory] = [
                    str(Path(anomaly_mask_path, mask))
                    for mask in sorted(listdir(anomaly_mask_path))
                ]
            else:
                mask_paths["good"] = None

            data_tuples = []
            for i, img_path in enumerate(img_paths[subcategory]):
                data_tuple = [img_path, None]
                if subcategory != "good":
                    data_tuple[1] = mask_paths[subcategory][i]
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

            self.data_list.extend(data_tuples)

        self.imgs = []
        self.masks = []
        for img_path, mask_path in self.data_list:
            self.imgs.append(self.transform_img(open(img_path).convert("RGB")))

            if mask_path is not None:
                mask = self.transform_mask(open(mask_path))
            else:
                mask = zeros([1, *self.imgs[-1].size()[1:]])

            self.masks.append(mask.int())

        self.imgs = stack(self.imgs)
        self.masks = stack(self.masks)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i: int):
        return self.imgs[i], self.masks[i]


class MVTecDataModule(LightningDataModule):
    def __init__(
        self, dataset_path: str, img_size: int, batch_size: int, val_ratio: float
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        transforms = [
            Resize(img_size),
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
            self.dataset_path,
            self.transform_img,
        )
        self.val_dataset = MVTecDataset(
            "val",
            self.dataset_path,
            self.transform_img,
            self.transform_mask,
            self.val_ratio,
        )
        self.test_dataset = MVTecDataset(
            "test",
            self.dataset_path,
            self.transform_img,
            self.transform_mask,
            self.val_ratio,
        )

        return self

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
