"""
Mixed Points Training Script - Baseline with Positive AND Negative Points
This script samples both positive points (inside mask) and negative points (outside mask)
to help the model better learn boundaries and improve segmentation accuracy.
"""
import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json
from datetime import datetime

def set_seeds():
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def read_batch(data, neg_to_pos_ratio=0.5, visualize_data=False):
    """
    Read a batch with both positive and negative points.

    Args:
        data: Training data list
        neg_to_pos_ratio: Ratio of negative to positive points (default 0.5 means half as many negative as positive)
        visualize_data: Whether to visualize the sampled points

    Returns:
        Img: RGB image
        binary_mask: Binary segmentation mask
        points: Sampled points (both positive and negative)
        num_objects: Number of objects in the mask
    """
    # Select a random entry
    ent = data[np.random.randint(len(data))]

    # Get full paths
    Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)

    if Img is None or ann_map is None:
        print(f"Error: Could not read image or mask")
        return None, None, None, None, 0

    # Resize image and mask
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []
    labels = []

    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask)

    # Erode the combined binary mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    # Sample POSITIVE points (inside the mask)
    coords_positive = np.argwhere(eroded_mask > 0)
    if len(coords_positive) > 0:
        num_positive = len(inds)
        for _ in range(num_positive):
            yx = np.array(coords_positive[np.random.randint(len(coords_positive))])
            points.append([yx[1], yx[0]])
            labels.append(1)  # Positive label

        # Sample NEGATIVE points (outside the mask)
        # Create a mask for background regions (not in binary_mask)
        background_mask = (binary_mask == 0).astype(np.uint8)
        # Erode background slightly to avoid border regions
        eroded_background = cv2.erode(background_mask, np.ones((5, 5), np.uint8), iterations=1)
        coords_negative = np.argwhere(eroded_background > 0)

        if len(coords_negative) > 0:
            num_negative = max(1, int(num_positive * neg_to_pos_ratio))
            for _ in range(num_negative):
                yx = np.array(coords_negative[np.random.randint(len(coords_negative))])
                points.append([yx[1], yx[0]])
                labels.append(0)  # Negative label

    if len(points) == 0:
        return None, None, None, None, 0

    points = np.array(points)
    labels = np.array(labels)

    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)
    labels = np.expand_dims(labels, axis=1)

    return Img, binary_mask, points, labels, len(inds)

class ConvergenceDetector:
    """Detect convergence based on IoU plateau"""
    def __init__(self, patience=1000, min_delta=0.001, check_window=500):
        self.patience = patience
        self.min_delta = min_delta
        self.check_window = check_window
        self.best_iou = 0
        self.steps_without_improvement = 0
        self.iou_history = []

    def update(self, current_iou, step):
        self.iou_history.append(current_iou)

        # Only check every check_window steps
        if step % self.check_window != 0:
            return False

        # Need at least check_window samples
        if len(self.iou_history) < self.check_window:
            return False

        # Average IoU over last check_window steps
        recent_avg = np.mean(self.iou_history[-self.check_window:])

        # Check if improved
        if recent_avg > self.best_iou + self.min_delta:
            self.best_iou = recent_avg
            self.steps_without_improvement = 0
            print(f"  [Convergence] New best IoU: {self.best_iou:.6f}")
            return False
        else:
            self.steps_without_improvement += self.check_window
            print(f"  [Convergence] No improvement for {self.steps_without_improvement} steps (best: {self.best_iou:.6f})")

            if self.steps_without_improvement >= self.patience:
                print(f"\n{'='*70}")
                print(f"CONVERGENCE DETECTED!")
                print(f"Best IoU: {self.best_iou:.6f}")
                print(f"No improvement for {self.steps_without_improvement} steps")
                print(f"{'='*70}\n")
                return True

        return False

def main():
    print("="*70)
    print("MIXED POINTS TRAINING - POSITIVE + NEGATIVE POINTS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seeds()

    # Training hyperparameters
    NEG_TO_POS_RATIO = 0.5  # Sample 50% as many negative points as positive points

    print(f"\nTraining configuration:")
    print(f"  Negative to Positive ratio: {NEG_TO_POS_RATIO}")

    # Load dataset
    print("\nLoading dataset...")
    data_dir = "../../../leaf-seg/leaf-seg"
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_data = []
    for index, row in train_df.iterrows():
        train_data.append({
            "image": os.path.join(images_dir, row['imageid']),
            "annotation": os.path.join(masks_dir, row['maskid'])
        })

    test_data = []
    for index, row in test_df.iterrows():
        test_data.append({
            "image": os.path.join(images_dir, row['imageid']),
            "annotation": os.path.join(masks_dir, row['maskid'])
        })

    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Load SAM2 model
    print("\nLoading SAM2 model...")
    sam2_checkpoint = "../../../../sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Train mask decoder and prompt encoder (layer-wise fine-tuning)
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Mixed precision
    scaler = torch.amp.GradScaler()

    # Training parameters
    MAX_STEPS = 20000
    FINE_TUNED_MODEL_NAME = "mixed_points_sam2"
    MODEL_SAVE_DIR = "../../models/mixed_points"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Configure optimizer
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.6)

    # Gradient accumulation
    accumulation_steps = 4

    # Convergence detection
    convergence_detector = ConvergenceDetector(
        patience=2000,
        min_delta=0.001,
        check_window=500
    )

    mean_iou = 0
    training_log = []

    print(f"\nStarting mixed points training...")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Convergence patience: 2000 steps")
    print(f"Check window: 500 steps")
    print("="*70)

    for step in range(1, MAX_STEPS + 1):
        with torch.amp.autocast(device_type='cuda'):
            image, mask, input_point, input_label, num_masks = read_batch(
                train_data,
                neg_to_pos_ratio=NEG_TO_POS_RATIO,
                visualize_data=False
            )

            if image is None or mask is None or num_masks == 0:
                continue

            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) -
                       (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            # Apply gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                predictor.model.zero_grad()

            if step % 500 == 0:
                FINE_TUNED_MODEL = os.path.join(MODEL_SAVE_DIR, FINE_TUNED_MODEL_NAME + "_" + str(step) + ".pt")
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

            if step == 1:
                mean_iou = 0

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            current_lr = optimizer.param_groups[0]["lr"]

            if step % 100 == 0:
                print(f"Step {step}: LR = {current_lr:.6f}, IoU = {mean_iou:.6f}, Seg Loss = {seg_loss:.6f}")

                # Log training progress
                training_log.append({
                    'step': step,
                    'lr': current_lr,
                    'iou': mean_iou,
                    'seg_loss': seg_loss.item()
                })

            # Check convergence (commented out for now - run full MAX_STEPS)
            # if convergence_detector.update(mean_iou, step):
            #     print(f"\nTraining converged at step {step}")
            #     FINAL_MODEL = os.path.join(MODEL_SAVE_DIR, FINE_TUNED_MODEL_NAME + f"_final_{step}.pt")
            #     torch.save(predictor.model.state_dict(), FINAL_MODEL)
            #     print(f"Final model saved: {FINAL_MODEL}")
            #     break

    # Save final model if reached max steps
    if step == MAX_STEPS:
        print(f"\nReached maximum steps: {MAX_STEPS}")
        FINAL_MODEL = os.path.join(MODEL_SAVE_DIR, FINE_TUNED_MODEL_NAME + f"_final_{step}.pt")
        torch.save(predictor.model.state_dict(), FINAL_MODEL)
        print(f"Final model saved: {FINAL_MODEL}")

    # Save training log
    log_file = "../../results/mixed_points_training_log.json"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump({
            'final_step': step,
            'final_iou': mean_iou,
            'best_iou': convergence_detector.best_iou,
            'neg_to_pos_ratio': NEG_TO_POS_RATIO,
            'training_log': training_log
        }, f, indent=2)

    print(f"\nTraining log saved: {log_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("MIXED POINTS TRAINING COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
