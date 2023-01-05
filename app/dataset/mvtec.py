from torch.utils.data import Dataset
import cv2
from albumentations import Compose
from pathlib import Path


class MVTec(Dataset):
    def __init__(self, category: str, work_dir: str, transform: Compose):
        self.transform = transform
        self.imgs = []

        # loop through all paths in file img_list.txt
        img_path = Path(work_dir, 'MVTec', 'img_list.txt')
        with open(img_path, 'r') as f:
            self.imgs.extend(str(Path(work_dir, 'MVTec', line.strip())) for line in f)
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)