from os import listdir
from torch import zeros
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pathlib import Path
from PIL.Image import open


class MVTec(Dataset):
    def __init__(
        self,
        split: str,
        work_dir: str,
        dataset_dir: str,
        category: str,
        holdout_ratio: float,
        transform_img: Compose,
        transform_mask: Compose = None,
        holdout: bool = False,
    ):
        self.split = split
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        dataset_path = Path(work_dir, dataset_dir)
        category_path = Path(dataset_path, category)
        split_path = Path(category_path, split if split != "val" else "train")
        subcategories = listdir(split_path)

        img_paths = {}
        mask_paths = {}

        self.data = []

        for subcategory in subcategories:
            subcategory_path = Path(split_path, subcategory)
            img_paths[subcategory] = [
                str(Path(subcategory_path, img))
                for img in sorted(listdir(subcategory_path))
            ]

            if split == "test" and subcategory != "good":
                anomaly_mask_path = Path(category_path, "ground_truth", subcategory)
                mask_paths[subcategory] = [
                    str(Path(anomaly_mask_path, mask))
                    for mask in sorted(listdir(anomaly_mask_path))
                ]
            else:
                mask_paths["good"] = None

            data_tuples = []
            for i, img_path in enumerate(img_paths[subcategory]):
                data_tuple = [img_path, None]
                if self.split == "test" and subcategory != "good":
                    data_tuple[1] = mask_paths[subcategory][i]
                data_tuples.append(data_tuple)
                
            split_index = int(holdout_ratio * len(data_tuples))
            if holdout:
                data_tuples = data_tuples[split_index:]
            else:
                data_tuples = data_tuples[:split_index]
                
            self.data.extend(data_tuples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.data[idx]
        img = self.transform_img(open(img_path).convert("RGB"))

        if self.split == "test" and mask_path is not None:
            mask = self.transform_mask(open(mask_path))
        else:
            mask = zeros([1, *img.size()[1:]])

        return {"img": img, "mask": mask}
