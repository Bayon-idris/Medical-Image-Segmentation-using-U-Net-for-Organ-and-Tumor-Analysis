import os
import torch
from torch.utils.data import Dataset
from preprocessing import preprocess_case


class HeartDataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_size=256):
        self.cases = []
        self.target_size = target_size

        for fname in os.listdir(images_dir):
            if fname.endswith(".nii.gz"):
                self.cases.append(
                    (
                        os.path.join(images_dir, fname),
                        os.path.join(labels_dir, fname.replace("_0000", "")),
                    )
                )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        image_path, label_path = self.cases[idx]

        slices = preprocess_case(image_path, label_path, self.target_size)

        image, mask = slices[len(slices) // 2]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
