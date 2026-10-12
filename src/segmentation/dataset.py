from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ProstateSegmentationDataset(Dataset):
    """
    PyTorch dataset for 2D prostate segmentation.
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        splits_csv: str | Path,
        split: str,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        split_data = pd.read_csv(
            splits_csv,
            dtype={
                "patient_id": str,
                "study_id": str,
            },
        )

        split_data = split_data[
            split_data["split"] == split
        ]

        valid_case_ids = {
            f"{row.patient_id}_{row.study_id}"
            for row in split_data.itertuples()
        }

        self.samples = []

        for image_path in sorted(
            self.images_dir.glob("*.npy")
        ):
            filename = image_path.name

            case_id = "_".join(
                filename.split("_")[:2]
            )

            if case_id not in valid_case_ids:
                continue

            mask_path = self.masks_dir / filename

            if mask_path.exists():
                self.samples.append(
                    (image_path, mask_path)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        image = np.load(
            image_path
        ).astype(np.float32)

        mask = np.load(
            mask_path
        ).astype(np.float32)

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask