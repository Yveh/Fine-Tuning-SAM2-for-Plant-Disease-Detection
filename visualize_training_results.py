"""
Visualize Training Results Comparison
Generates 3 plots in one row:
1. Test IoU with error bars for each method
2. Train vs Test IoU comparison
3. Generalization Gap (Train IoU - Test IoU)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_training_log(log_path):
    """Load training log JSON file"""
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return None

    with open(log_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_comparison(methods_data, output_path='../results/training_comparison.png'):
    """
    Visualize training results comparison

    Args:
        methods_data: List of dicts, each containing:
            {
                'name': str,  # Method name for display
                'train_iou': float,  # Final training IoU
                'test_iou_mean': float,  # Mean test IoU
                'test_iou_std': float,  # Std of test IoU
                'color': str  # Color for plotting (optional)
            }
        output_path: Path to save the figure
    """

    # Extract data
    method_names = [m['name'] for m in methods_data]
    train_ious = [m['train_iou'] for m in methods_data]
    test_ious_mean = [m['test_iou_mean'] for m in methods_data]
    test_ious_std = [m['test_iou_std'] for m in methods_data]

    # Calculate generalization gap
    gen_gaps = [train - test for train, test in zip(train_ious, test_ious_mean)]

    # Set colors
    colors = [m.get('color', f'C{i}') for i, m in enumerate(methods_data)]

    # Create figure with 3 subplots in one row
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x_pos = np.arange(len(method_names))

    # Plot 1: Test IoU with error bars
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, test_ious_mean, yerr=test_ious_std,
                    capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test IoU', fontsize=12, fontweight='bold')
    ax1.set_title('Test IoU Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.12)  # Increased to accommodate labels above error bars

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, test_ious_mean, test_ious_std)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Train vs Test IoU
    ax2 = axes[1]
    width = 0.35
    x_pos_offset = x_pos - width/2

    bars2a = ax2.bar(x_pos_offset, train_ious, width, label='Train IoU',
                     alpha=0.8, color='skyblue', edgecolor='black', linewidth=1.5)
    bars2b = ax2.bar(x_pos_offset + width, test_ious_mean, width, label='Test IoU',
                     alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('IoU', fontsize=12, fontweight='bold')
    ax2.set_title('Train vs Test IoU', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.08)  # Increased to accommodate labels above bars

    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # Plot 3: Generalization Gap
    ax3 = axes[2]
    bars3 = ax3.bar(x_pos, gen_gaps, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Generalization Gap', fontsize=12, fontweight='bold')
    ax3.set_title('Generalization Gap (Train - Test)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_names, rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 0.1)  # Increased to accommodate labels above bars

    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars3, gen_gaps)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005 if height >= 0 else height - 0.01,
                f'{gap:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")

    plt.close()

def main():
    """
    Main function: Configure your methods data here
    """

    # Example: Configure your methods data
    # Each method should have: name, train_iou, test_iou_mean, test_iou_std
    methods_data = [
        {
            'name': 'Baseline',
            'train_iou': 0.6945313188159298,  # Replace with actual value from training log
            'test_iou_mean': 0.6387705070867326,  # Replace with actual value from evaluation
            'test_iou_std': 0.19839120390108514,  # Replace with actual std from evaluation
            'color': '#3498db'  # Blue
        },
        # {
        #     'name': 'Data Aug',
        #     'train_iou': 0.6544683533538841,  # From checkpoint evaluation
        #     'test_iou_mean': 0.6492824361411788,  # Replace with actual value
        #     'test_iou_std': 0.19357427649553388,  # Replace with actual std
        #     'color': '#2ecc71'  # Green
        # },
        # {
        #     'name': 'Mixed Points Prompt',
        #     'train_iou': 0.7056244015693665,  # From checkpoint evaluation
        #     'test_iou_mean': 0.6240983782651275,  # Replace with actual value
        #     'test_iou_std': 0.2013949747260709,  # Replace with actual std
        #     'color': '#9467bd'  # Purple
        # },
        # {
        #     'name': 'Box Prompt',
        #     'train_iou': 0.6833606688405729,  # From checkpoint evaluation
        #     'test_iou_mean': 0.6360894858428556,  # Replace with actual value
        #     'test_iou_std': 0.23741257058336027,  # Replace with actual std
        #     'color': '#e74c3c'  # Red
        # },
        # {
        #     'name': 'Hybrid Prompt',
        #     'train_iou': 0.6910973756962541,  # From checkpoint evaluation
        #     'test_iou_mean': 0.6754387351527074,  # Replace with actual value
        #     'test_iou_std': 0.20501047589766624,  # Replace with actual std
        #     'color': '#f39c12'  # Orange
        # },
        {
            'name': 'Ours',
            'train_iou': 0.736691,  # From checkpoint evaluation
            'test_iou_mean': 0.6958971498965013,  # Replace with actual value
            'test_iou_std': 0.20676150695536538,  # Replace with actual std
            'color': '#db3498'  # Orange
        },
        
    ]

    # Optional: Load from training logs
    # Example of loading training IoU from log file:
    # baseline_log = load_training_log('../results/baseline_training_log.json')
    # if baseline_log:
    #     methods_data[0]['train_iou'] = baseline_log['final_iou']

    print("="*70)
    print("Generating Training Results Comparison")
    print("="*70)

    print("\nMethods to compare:")
    for m in methods_data:
        print(f"  - {m['name']}: Train IoU={m['train_iou']:.3f}, "
              f"Test IoU={m['test_iou_mean']:.3f}±{m['test_iou_std']:.3f}")

    # Generate visualization
    output_path = '../results/training_comparison.png'
    visualize_comparison(methods_data, output_path)

    print(f"\n{'='*70}")
    print("Comparison visualization completed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
