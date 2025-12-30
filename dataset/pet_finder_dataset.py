import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class PetFinderDataset(Dataset):
    def __init__(self, csv_path, is_train, transform=None):
        self.root = os.path.dirname(csv_path)
        if is_train:
            self.root = os.path.join(self.root, "train")
        else:
            self.root = os.path.join(self.root, "test")

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, f"{self.df.iloc[idx, 0]}.jpg")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0


#
