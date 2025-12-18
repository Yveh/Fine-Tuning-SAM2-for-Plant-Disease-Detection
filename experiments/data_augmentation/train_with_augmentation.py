"""
Training with Data Augmentation - Enhanced Baseline
Includes: random flips, rotations, scaling, brightness adjustments, and color jittering
"""

import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
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

import albumentations as A

import logging
logging.getLogger().setLevel(logging.WARNING)

def build_train_aug(h, w):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 旋转（-45~45)
        A.Rotate(limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT, p=0.7),

        # 缩放（±10%）
        A.RandomScale(scale_limit=0.1, interpolation=cv2.INTER_LINEAR, p=0.7),

        # 保证大小不变：缩放后可能变小/变大，这里补齐再裁回原尺寸
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.RandomCrop(height=h, width=w, p=1.0),

        # 亮度/对比度
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        # Hue/Saturation
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, p=0.3),
    ])

def apply_augmentation_albu(image, mask):
    h, w = image.shape[:2]
    aug = build_train_aug(h, w)
    out = aug(image=image, mask=mask)
    return out["image"], out["mask"]


def read_batch(data, use_augmentation=True):
    """Read a batch with optional data augmentation"""
    ent = data[np.random.randint(len(data))]

    # Read image and mask
    Img = cv2.imread(ent["image"])[..., ::-1]  # BGR to RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)

    if Img is None or ann_map is None:
        return None, None, None, 0

    # Resize
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # Apply data augmentation (only during training)
    if use_augmentation:
        Img, ann_map = apply_augmentation_albu(Img, ann_map)

    # Create binary mask and sample points
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []

    inds = np.unique(ann_map)[1:]  # Skip background
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask)

    # Erode and sample points
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded_mask > 0)

    if len(coords) > 0:
        for _ in inds:
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])

    points = np.array(points)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    return Img, binary_mask, points, len(inds)

class ConvergenceDetector:
    """Detect convergence based on IoU plateau"""
    def __init__(self, patience=2000, min_delta=0.001, check_window=500):
        self.patience = patience
        self.min_delta = min_delta
        self.check_window = check_window
        self.best_iou = 0
        self.steps_without_improvement = 0
        self.iou_history = []

    def update(self, current_iou, step):
        self.iou_history.append(current_iou)

        if step % self.check_window != 0:
            return False

        if len(self.iou_history) < self.check_window:
            return False

        recent_avg = np.mean(self.iou_history[-self.check_window:])

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
    print("TRAINING WITH DATA AUGMENTATION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seeds()

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

    print(f"Training samples: {len(train_data)}")

    # Load SAM2 model
    print("\nLoading SAM2 model...")
    sam2_checkpoint = "../../../../sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Layer-wise fine-tuning
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Mixed precision
    scaler = torch.amp.GradScaler()

    # Training parameters
    MAX_STEPS = 40000
    FINE_TUNED_MODEL_NAME = "augmented_sam2"
    MODEL_SAVE_DIR = "../../models/data_augmentation"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Optimizer and scheduler - same as baseline
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.6)

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

    print(f"\nStarting training with data augmentation...")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Convergence patience: 2000 steps")
    print("="*70)

    for step in range(1, MAX_STEPS + 1):
        with torch.amp.autocast(device_type='cuda'):
            # Read batch WITH augmentation
            image, mask, input_point, num_masks = read_batch(train_data, use_augmentation=True)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
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

            # Gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                predictor.model.zero_grad()

            # Save checkpoints
            if step % 500 == 0:
                FINE_TUNED_MODEL = os.path.join(MODEL_SAVE_DIR, FINE_TUNED_MODEL_NAME + "_" + str(step) + ".pt")
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

            if step == 1:
                mean_iou = 0

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            current_lr = optimizer.param_groups[0]["lr"]

            if step % 100 == 0:
                print(f"Step {step}: LR = {current_lr:.6f}, IoU = {mean_iou:.6f}, Seg Loss = {seg_loss:.6f}")

                training_log.append({
                    'step': step,
                    'lr': current_lr,
                    'iou': mean_iou,
                    'seg_loss': seg_loss.item()
                })

            # # Check convergence
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
    log_file = "../../results/augmented_training_log.json"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump({
            'final_step': step,
            'final_iou': mean_iou,
            'best_iou': convergence_detector.best_iou,
            'training_log': training_log,
            'augmentation_config': {
                'horizontal_flip': True,
                'vertical_flip': True,
                'rotation_range': [-15, 15],
                'scale_range': [0.9, 1.1],
                'brightness_range': [0.7, 1.3],
                'contrast_range': [0.8, 1.2],
                'hue_shift_range': [-10, 10],
                'saturation_range': [0.7, 1.3]
            }
        }, f, indent=2)

    print(f"\nTraining log saved: {log_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("TRAINING WITH DATA AUGMENTATION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
