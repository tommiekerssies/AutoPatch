from torch.utils.data import Dataset
import cv2
import os
from albumentations import Compose


class AOI(Dataset):
    def __init__(self, dataset_split_path: str, transform: Compose):
        self.imgs = []
        self.masks = []
        self.transform = transform

        for img in os.listdir(os.path.join(dataset_split_path, "img")):
            self.imgs.append(os.path.join(dataset_split_path, "img", img))
            self.masks.append(
                os.path.join(
                    dataset_split_path, "lbl", f"{os.path.splitext(img)[0]}.png"
                )
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masks = [mask.copy(), mask.copy(), mask.copy()]

        sample = self.transform(image=img, masks=masks, ignore_mask=mask.copy())

        # TODO: do this based on number of classes and don't hardcode
        self.make_mask_binary(sample["masks"][0], [1, 3, 5, 7])
        self.make_mask_binary(sample["masks"][1], [2, 3, 6, 7])
        self.make_mask_binary(sample["masks"][2], [4, 5, 6, 7])
        self.make_mask_binary(sample["ignore_mask"], [254])
        
        sample["img_path"] = self.imgs[idx]
        sample["mask_path"] = self.masks[idx]

        return sample

    @staticmethod
    def make_mask_binary(mask, class_ids):
        if 1 not in class_ids:
            mask[mask == 1] = 0

        for class_id in class_ids:
            mask[mask == class_id] = 1

        mask[mask != 1] = 0
