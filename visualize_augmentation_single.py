"""
Generate Individual Augmented Images with Mask Overlay (No Captions)
Each image is saved as a separate file
"""

import os
import random
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def build_train_aug(h, w):
    """Data augmentation pipeline from training script"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Rotation (-45~45 degrees)
        A.Rotate(limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT, p=0.7),
        # Random scale (±10%)
        A.RandomScale(scale_limit=0.1, interpolation=cv2.INTER_LINEAR, p=0.7),
        # Pad and crop to maintain size
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.RandomCrop(height=h, width=w, p=1.0),
        # Brightness/Contrast
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # Hue/Saturation
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, p=0.3),
    ])

def apply_augmentation(image, mask):
    """Apply augmentation to image and mask"""
    h, w = image.shape[:2]
    aug = build_train_aug(h, w)
    out = aug(image=image, mask=mask)
    return out["image"], out["mask"]

def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay colored mask on image"""
    # Create a colored version of the mask (green)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]  # Green

    # Blend
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay

def save_single_image(image, output_path):
    """Save a single image without any captions or axes"""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(image.shape[1]/100, image.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    print("="*70)
    print("Generating Individual Augmented Images (with Mask)")
    print("="*70)

    set_seeds(42)

    # Load dataset
    print("\nLoading dataset...")
    data_dir = "../../leaf-seg/leaf-seg"
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df, _ = train_test_split(train_df, test_size=0.2, random_state=42)

    # Create output directory
    output_dir = "../results/augmentation_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual images
    num_samples = 10
    num_augmentations_per_sample = 9

    print(f"\nGenerating {num_samples} samples with {num_augmentations_per_sample} augmentations each...")

    for sample_idx in range(num_samples):
        # Select a random image
        row = train_df.iloc[np.random.randint(len(train_df))]
        image_path = os.path.join(images_dir, row['imageid'])
        mask_path = os.path.join(masks_dir, row['maskid'])

        # Read image and mask
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"  Warning: Could not read sample {sample_idx+1}, skipping...")
            continue

        # Convert BGR to RGB
        img = img[..., ::-1]

        # Resize to reasonable size for visualization
        max_size = 512
        r = min(max_size / img.shape[1], max_size / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

        # Create binary mask
        binary_mask = (mask > 0).astype(np.uint8) * 255

        print(f"  Sample {sample_idx+1}/{num_samples}:")

        # Save original image with mask
        original_overlay = overlay_mask_on_image(img, binary_mask)
        output_path = os.path.join(output_dir, f"sample_{sample_idx+1:02d}_original.png")
        save_single_image(original_overlay, output_path)
        print(f"    Saved: sample_{sample_idx+1:02d}_original.png")

        # Generate and save augmented versions
        for aug_idx in range(num_augmentations_per_sample):
            aug_img, aug_mask = apply_augmentation(img.copy(), binary_mask.copy())
            aug_overlay = overlay_mask_on_image(aug_img, aug_mask)

            output_path = os.path.join(output_dir, f"sample_{sample_idx+1:02d}_aug_{aug_idx+1:02d}.png")
            save_single_image(aug_overlay, output_path)

        print(f"    Saved: sample_{sample_idx+1:02d}_aug_01 to aug_{num_augmentations_per_sample:02d}.png")

    print(f"\n{'='*70}")
    print(f"All individual images saved to: {output_dir}")
    print(f"Total files: {num_samples * (1 + num_augmentations_per_sample)} images")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
