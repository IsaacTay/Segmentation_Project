from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from PIL import Image
from pathlib import Path

class BusiDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None, split="train"):
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)
        mask_paths = list(root.rglob("*_mask.png"))
        train_paths, tmp_paths = train_test_split(mask_paths, test_size=0.2, random_state=0)
        if split != "train":
          val_paths, test_paths = train_test_split(tmp_paths, test_size=0.5, random_state=0)
          self.mask_paths = val_paths if split == "val" else test_paths
        else:
          self.mask_paths = train_paths

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        img_path = Path(str(mask_path)[:-9] + ".png")
        mask = Image.open(mask_path)
        img = Image.open(img_path) ## TODO: make the image mono to reduct compute cost

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.mask_paths)
