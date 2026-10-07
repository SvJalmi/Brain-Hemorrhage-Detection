from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class MBHSeg25Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.cases = sorted(
            [
                p for p in self.root_dir.iterdir()
                if p.is_dir() and (p / "image.nii.gz").exists()
            ]
        )

        if not self.cases:
            raise RuntimeError(
                f"No MBH-Seg25 cases found in {self.root_dir}"
            )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case_dir = self.cases[index]

        image_path = case_dir / "image.nii.gz"

        annotation_paths = sorted(
            case_dir.glob("label_annot_*.nii.gz")
        )

        if not annotation_paths:
            raise RuntimeError(
                f"No annotations found for {case_dir.name}"
            )

        image_nii = nib.load(str(image_path))
        image = image_nii.get_fdata(dtype=np.float32)

        annotations = []

        for annotation_path in annotation_paths:
            annotation_nii = nib.load(str(annotation_path))
            mask = annotation_nii.get_fdata(dtype=np.float32)

            if mask.shape != image.shape:
                raise RuntimeError(
                    f"Shape mismatch in {case_dir.name}: "
                    f"image={image.shape}, "
                    f"mask={mask.shape}"
                )

            annotations.append(mask.astype(np.uint8))

        return {
            "case_id": case_dir.name,
            "image": image,
            "annotations": annotations,
            "image_affine": image_nii.affine,
            "image_spacing": image_nii.header.get_zooms()[:3],
        }