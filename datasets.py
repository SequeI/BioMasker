import os
import numpy as np
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    def __init__(self, root_dir="data", phase="warmup", split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / phase / "img" / split
        self.ann_dir = self.root_dir / phase / "ann" / split
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_dir / self.img_list[idx]))
        ann_path = self.ann_dir / f"{Path(self.img_list[idx]).stem}.png"
        ann = np.array(Image.open(ann_path))

        if self.transform:
            augmented = self.transform(image=img, mask=ann)
            img = augmented["image"]
            ann = augmented["mask"]

        return img, ann


class SegDataModule(pl.LightningDataModule):
    def __init__(self, root_dir="data", phase="warmup", batch_size: int = 8):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="train",
            transform=A.Compose([A.RandomCrop(380, 380), ToTensorV2()]),
        )
        self.valid = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="valid",
            transform=ToTensorV2()
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False)


if __name__ == "__main__":
    ds = SegDataset()
    img, ann = ds[10]
    print("Done!")