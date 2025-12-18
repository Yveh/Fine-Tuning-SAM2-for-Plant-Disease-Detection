"""
Hybrid Prompt SAM2 Evaluation with BOTH Points and Box Prompts

Call:
    metrics = evaluate_sam2_model(
        model_path="../../models/hybrid_prompt/hybrid_prompt_sam2_final_20000.pt",
        data_dir="../../../leaf-seg/leaf-seg",
        sam2_checkpoint="../../../../sam2_hiera_tiny.pt",
        model_cfg="sam2_hiera_t.yaml",
        test_size=0.2,
        seed=42,
        device="cuda",
    )
    print(metrics)

This evaluation uses HYBRID prompts (both points AND boxes):
- extract bounding box from GT (eroded mask)
- sample one point from inside the mask
- predictor.predict(point_coords=..., point_labels=..., box=..., multimask_output=True)
- choose best mask by score (argmax)
- compute binary IoU + Dice against GT
"""

import os
import sys
import random
from typing import Dict, Optional, List

import pandas as pd
import cv2
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Adjust this if your relative import root differs
sys.path.insert(0, "../..")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -------------------------
# Utilities
# -------------------------
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_image(image_path: str, mask_path: str):
    """Read + resize image/mask to max side 1024 (same as your baseline script)."""
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        return None, None

    img = img[..., ::-1]  # BGR -> RGB
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),
                      interpolation=cv2.INTER_NEAREST)
    return img, mask


def get_bounding_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract bounding box from binary mask.
    Returns box in xyxy format: [x_min, y_min, x_max, y_max]
    """
    # Optionally erode the mask to get a tighter box (similar to point sampling logic)
    eroded_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    coords = np.argwhere(eroded_mask > 0)
    if len(coords) == 0:
        # Fallback to original mask if erosion removes everything
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return np.array([x_min, y_min, x_max, y_max])


def sample_point_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Sample one random point from inside the mask.
    Returns point in xy format: [x, y]
    """
    # Erode the mask to avoid boundary points
    eroded_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    coords = np.argwhere(eroded_mask > 0)
    if len(coords) == 0:
        # Fallback to original mask if erosion removes everything
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None

    # Select one random point
    yx = coords[np.random.randint(len(coords))]
    # Return in xy format
    return np.array([yx[1], yx[0]])


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Binary IoU + Dice between predicted mask and GT."""
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = (pred_binary & gt_binary).sum()
    union = (pred_binary | gt_binary).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    denom = pred_binary.sum() + gt_binary.sum()
    dice = float(2 * intersection / denom) if denom > 0 else 0.0
    return iou, dice


def _normalize_predict_output(masks: np.ndarray, scores: np.ndarray):
    """
    SAM2 predictor outputs can vary by version:
      - masks: (M,H,W) or (1,M,H,W) or (Npts,M,H,W) depending on prompting/batching
      - scores: (M,) or (1,M) ...
    This function handles the common cases in your scripts.
    """
    # If batched: take first batch
    if masks is None or scores is None:
        return None, None

    if hasattr(masks, "ndim") and masks.ndim == 4:
        # (B, M, H, W) -> (M, H, W)
        masks = masks[0]
    if hasattr(scores, "ndim") and scores.ndim == 2:
        # (B, M) -> (M,)
        scores = scores[0]

    # At this point, expect masks: (M,H,W), scores: (M,)
    if not (hasattr(masks, "ndim") and masks.ndim == 3):
        return None, None
    if not (hasattr(scores, "ndim") and scores.ndim == 1):
        return None, None

    return masks, scores


# -------------------------
# Main evaluation function
# -------------------------
def evaluate_sam2_model(
    model_path: str,
    data_dir: str,
    sam2_checkpoint: str,
    model_cfg: str = "sam2_hiera_t.yaml",
    test_size: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
    max_samples: Optional[int] = None,  # for quick debug
) -> Dict[str, float]:
    """
    Evaluate any SAM2 checkpoint on leaf-seg test split using HYBRID prompts (points + boxes).

    Returns summary metrics only (mean/std IoU & Dice).
    """

    # ---- sanity checks ----
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    if not os.path.exists(sam2_checkpoint):
        raise FileNotFoundError(f"SAM2 base checkpoint not found: {sam2_checkpoint}")
    train_csv = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found at: {train_csv}")

    set_seeds(seed)

    # ---- build test split ----
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    df = pd.read_csv(train_csv)
    _, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    test_data = [
        {
            "image": os.path.join(images_dir, row["imageid"]),
            "annotation": os.path.join(masks_dir, row["maskid"]),
        }
        for _, row in test_df.iterrows()
    ]

    if max_samples is not None:
        test_data = test_data[:max_samples]

    # ---- load model ----
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    state = torch.load(model_path, map_location=device)
    predictor.model.load_state_dict(state)
    predictor.model.eval()

    # ---- evaluate ----
    ious: List[float] = []
    dices: List[float] = []

    for entry in tqdm(test_data, desc=f"Evaluating {os.path.basename(model_path)}"):
        try:
            image, gt_mask = read_image(entry["image"], entry["annotation"])
            if image is None or gt_mask is None:
                continue
            if gt_mask.sum() == 0:
                continue

            # Get bounding box from GT mask
            box = get_bounding_box(gt_mask)
            if box is None:
                continue

            # Sample one point from inside the mask
            point = sample_point_from_mask(gt_mask)
            if point is None:
                continue

            # Format prompts for SAM2
            # Box needs to be in format (1, 4) for SAM2
            box = box.reshape(1, 4)
            # Point needs to be in format (1, 2) for SAM2
            point_coords = point.reshape(1, 2)
            # Point label: 1 = foreground
            point_labels = np.array([1])

            with torch.no_grad():
                predictor.set_image(image)
                # Use BOTH point and box prompts
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True
                )

            masks, scores = _normalize_predict_output(masks, scores)
            if masks is None or scores is None:
                continue

            best_idx = int(np.argmax(scores))
            pred_mask = masks[best_idx]

            iou, dice = calculate_metrics(pred_mask, gt_mask)
            ious.append(iou)
            dices.append(dice)

        except Exception as e:
            print(f"[WARN] skipping sample due to error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {
        "model_path": model_path,
        "num_samples": float(len(ious)),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_dice": float(np.std(dices)) if dices else 0.0,
    }


# -------------------------
# Optional: simple entrypoint
# -------------------------
if __name__ == "__main__":
    # Example usage: edit paths as needed
    metrics = evaluate_sam2_model(
        # model_path="../models/hybrid_prompt/hybrid_prompt_sam2_4000.pt",
        model_path="../models/ours/ours_sam2_19500.pt",
        data_dir="../../leaf-seg/leaf-seg",
        sam2_checkpoint="../../../sam2_hiera_tiny.pt",
        model_cfg="sam2_hiera_t.yaml",
        test_size=0.2,
        seed=42,
        device="cuda",
        max_samples=None,  # set e.g. 50 for quick debug
    )
    print("\n=== Hybrid Prompt Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
