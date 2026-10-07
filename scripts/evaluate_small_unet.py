from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from src.segmentation.small_unet import SmallUNet, IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = PROJECT_ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"
CHECKPOINT = PROJECT_ROOT / "models" / "segmentation" / "small_unet_mbh_seg25_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}


def update_counts(pred, target, counts):
    for class_id in range(1, 6):
        p = pred == class_id
        t = target == class_id
        counts[class_id]["tp"] += int(np.logical_and(p, t).sum())
        counts[class_id]["fp"] += int(np.logical_and(p, np.logical_not(t)).sum())
        counts[class_id]["fn"] += int(np.logical_and(np.logical_not(p), t).sum())


def metric_from_counts(tp, fp, fn):
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    sensitivity_den = tp + fn
    precision_den = tp + fp
    dice = (2 * tp) / dice_den if dice_den > 0 else np.nan
    iou = tp / iou_den if iou_den > 0 else np.nan
    sensitivity = tp / sensitivity_den if sensitivity_den > 0 else np.nan
    precision = tp / precision_den if precision_den > 0 else np.nan
    return dice, iou, sensitivity, precision

def main():
    torch.set_num_threads(12)
    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(SPLIT_CSV)
    df = df[df["split"] == "test"].reset_index(drop=True)
    print("Test cases:", len(df))
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint validation loss:", checkpoint["val_loss"])

    counts = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(1, 6)}
    case_rows = []

    with torch.no_grad():
        for case_index, row in df.iterrows():
            image = nib.load(str(row["image_path"])).get_fdata(dtype=np.float32)
            target = nib.load(str(row["annot_3_path"])).get_fdata(dtype=np.float32).astype(np.uint8)
            case_counts = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(1, 6)}

            for z in range(image.shape[2]):
                image_slice = torch.from_numpy(image[:, :, z].copy()).unsqueeze(0).unsqueeze(0)
                image_slice = F.interpolate(image_slice, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
                image_slice = torch.clamp(image_slice, -1000, 1000)
                image_slice = (image_slice + 1000.0) / 2000.0
                logits = model(image_slice.to(device))
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                target_slice = torch.from_numpy(target[:, :, z].copy()).unsqueeze(0).unsqueeze(0).float()
                target_slice = F.interpolate(target_slice, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze().numpy().astype(np.uint8)
                update_counts(pred, target_slice, counts)
                update_counts(pred, target_slice, case_counts)

            row_result = {"case_id": row["case_id"]}
            for class_id, name in CLASS_NAMES.items():
                tp = case_counts[class_id]["tp"]
                fp = case_counts[class_id]["fp"]
                fn = case_counts[class_id]["fn"]
                dice, iou, sensitivity, precision = metric_from_counts(tp, fp, fn)
                row_result[f"{name}_dice"] = dice
                row_result[f"{name}_iou"] = iou
                row_result[f"{name}_sensitivity"] = sensitivity
                row_result[f"{name}_precision"] = precision
            case_rows.append(row_result)

            if (case_index + 1) % 5 == 0 or case_index == 0:
                print(f"Evaluated {case_index + 1}/{len(df)} test cases")

    rows = []
    for class_id, name in CLASS_NAMES.items():
        tp = counts[class_id]["tp"]
        fp = counts[class_id]["fp"]
        fn = counts[class_id]["fn"]
        dice, iou, sensitivity, precision = metric_from_counts(tp, fp, fn)
        rows.append({"class": name, "dice": dice, "iou": iou, "sensitivity": sensitivity, "precision": precision, "tp": tp, "fp": fp, "fn": fn})

    metrics_df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "small_unet_mbh_seg25_test_metrics.csv"
    case_path = RESULTS_DIR / "small_unet_mbh_seg25_test_case_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(case_rows).to_csv(case_path, index=False)

    print("\nTest metrics:")
    print(metrics_df.to_string(index=False))
    print("\nSaved:", metrics_path)
    print("Saved:", case_path)


if __name__ == "__main__":
    main()
