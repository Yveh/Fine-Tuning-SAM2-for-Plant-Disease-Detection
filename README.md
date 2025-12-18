# Fine-Tuning SAM2 for Plant Disease Detection

**CS550 Final Project**

## Overview

This project explores fine-tuning strategies for the Segment Anything Model 2 (SAM2) for plant disease detection and segmentation. We investigate different prompting strategies and training techniques to improve SAM2's performance on agricultural imagery.

## Project Structure

```
.
├── experiments/               # Different experimental approaches
│   ├── baseline/             # Standard SAM2 fine-tuning
│   ├── box_prompt/           # Box prompt-based training
│   ├── data_augmentation/    # Training with data augmentation
│   ├── hybrid_prompt/        # Combined prompting strategy
│   ├── mixed_points/         # Mixed point sampling
│   └── ours/                 # Our proposed method
├── evaluate.py               # Model evaluation script
├── evaluate_hybrid_prompt.py # Hybrid prompt evaluation
├── visualize_training_results.py    # Training results visualization
└── visualize_augmentation_single.py # Augmentation visualization
```

## Dataset

The project uses the [Leaf Segmentation Dataset](https://www.kaggle.com/c/leaf-segmentation) which contains:
- Plant leaf images with disease symptoms
- Pixel-level segmentation annotations
- Train/test split (80/20)

## Experimental Approaches

### 1. Baseline
Standard fine-tuning of SAM2 with point prompts sampled from the ground truth masks.

**Training Script**: `experiments/baseline/train_baseline.py`

### 2. Data Augmentation
Enhanced training with various augmentation techniques to improve model robustness.

**Training Script**: `experiments/data_augmentation/train_with_augmentation.py`

### 3. Mixed Points Prompt
Training with a mixture of positive and negative point prompts for better boundary detection.

**Training Script**: `experiments/mixed_points/train_mixed_points.py`

### 4. Box Prompt
Using bounding box prompts derived from ground truth masks to provide spatial context.

**Training Script**: `experiments/box_prompt/train_box_prompt.py`

### 5. Hybrid Prompt
Combining multiple prompt types (points + boxes) for comprehensive guidance.

**Training Script**: `experiments/hybrid_prompt/train_hybrid_prompt.py`

### 6. Ours (Proposed Method)
Our novel approach combining optimized prompting strategies and training techniques.

**Training Script**: `experiments/ours/train_ours.py`

## Requirements

### SAM2 Installation
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### Download SAM2 Checkpoint
Download the base SAM2 model checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_tiny.pt
```

## Usage

### Training

Each experiment has its own training script. To train a model:

```bash
# Example: Training the baseline model
cd experiments/baseline
python train_baseline.py

# Example: Training with our method
cd experiments/ours
python train_ours.py
```

### Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate.py
```


### Visualization

Visualize training results across different methods:

```bash
python visualize_training_results.py
```

This generates comparison plots showing:
- Test IoU with error bars for each method
- Train vs Test IoU comparison
- Generalization gap analysis

## References

1. Ravi, N., et al. (2024). "SAM 2: Segment Anything in Images and Videos." arXiv preprint arXiv:2408.00714.
2. Leaf Segmentation Dataset: https://www.kaggle.com/c/leaf-segmentation

## License

This project is for educational purposes as part of CS550 coursework.

## Acknowledgments

- SAM2 model from Meta's FAIR team
- Leaf segmentation dataset from Kaggle