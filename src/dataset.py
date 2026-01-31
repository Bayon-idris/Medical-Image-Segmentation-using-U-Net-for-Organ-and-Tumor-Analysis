import os
import torch
from torch.utils.data import Dataset
from preprocessing import preprocess_case

import numpy as np


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

        image, mask = slices[len(slices) // 2]  # prend une tranche

        # --- CORRECTION AXES ---
        # image : normalement [C,H,W] après preprocess_case
        if image.ndim == 4:  # [1,H,W,4] -> [1,H,W]
            image = image[..., 0]  # prends le premier canal

        if image.ndim == 2:  # [H,W] -> ajouter channel
            image = np.expand_dims(image, axis=0)  # [1,H,W]

        # mask : [H,W,num_classes] -> [H,W] (indice de classe)
        if mask.ndim == 3:  # [H,W,C]
            mask = np.argmax(mask, axis=-1)

        image = torch.tensor(image, dtype=torch.float32)  # [C,H,W]
        mask = torch.tensor(mask, dtype=torch.long)  # [H,W]

        return image, mask
