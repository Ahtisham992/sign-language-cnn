"""
Script to verify that label correction worked properly
Save as verify_correction.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def verify_label_correction():
    """Verify that the label correction mapping is working"""

    dataset_path = r"D:\Software enginner\university\sem7\GenAI\ASS-01\data\raw\sign-language-digits-dataset"

    x_file = os.path.join(dataset_path, 'X.npy')
    y_file = os.path.join(dataset_path, 'Y.npy')

    if not os.path.exists(x_file):
        x_file = os.path.join(dataset_path, 'x.npy')
        y_file = os.path.join(dataset_path, 'y.npy')

    X = np.load(x_file)
    Y = np.load(y_file)

    # Convert one-hot to class indices
    if len(Y.shape) == 2:
        original_labels = np.argmax(Y, axis=1)
    else:
        original_labels = Y

    # Apply the correction mapping
    label_correction_map = {
        0: 9, 1: 0, 2: 7, 3: 6, 4: 1,
        5: 8, 6: 4, 7: 6, 8: 2, 9: 5
    }

    corrected_labels = np.array([label_correction_map[label] for label in original_labels])

    # Show comparison
    fig, axes = plt.subplots(3, 10, figsize=(25, 12))
    fig.suptitle('Label Correction Verification', fontsize=16)

    # Row 1: Original labels (what the dataset claims)
    # Row 2: The actual images
    # Row 3: Corrected labels (what they should be)

    for digit in range(10):
        # Find first sample of original digit
        original_digit_indices = np.where(original_labels == digit)[0]

        if len(original_digit_indices) > 0:
            idx = original_digit_indices[0]
            img = X[idx]

            # Original label (incorrect)
            axes[0, digit].text(0.5, 0.5, f'Dataset says:\n{digit}',
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            axes[0, digit].set_title(f'Original Label: {digit}')
            axes[0, digit].axis('off')

            # The actual image
            axes[1, digit].imshow(img, cmap='gray')
            axes[1, digit].set_title(f'Actual Image (Index {idx})')
            axes[1, digit].axis('off')

            # Corrected label
            corrected_digit = corrected_labels[idx]
            axes[2, digit].text(0.5, 0.5, f'Should be:\n{corrected_digit}',
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[2, digit].set_title(f'Corrected Label: {corrected_digit}')
            axes[2, digit].axis('off')

            print(f"Original digit {digit} -> Corrected to {corrected_digit} (Index {idx})")

    plt.tight_layout()
    plt.show()

    # Show distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original distribution
    orig_counts = np.bincount(original_labels)
    ax1.bar(range(len(orig_counts)), orig_counts, color='lightcoral', alpha=0.7)
    ax1.set_title('Original Label Distribution')
    ax1.set_xlabel('Digit Class')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(10))

    # Corrected distribution
    corr_counts = np.bincount(corrected_labels)
    ax2.bar(range(len(corr_counts)), corr_counts, color='lightgreen', alpha=0.7)
    ax2.set_title('Corrected Label Distribution')
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(10))

    plt.tight_layout()
    plt.show()

    print("\nDistribution comparison:")
    print("Digit | Original | Corrected")
    print("------|----------|----------")
    for i in range(10):
        print(f"  {i}   |   {orig_counts[i]}    |    {corr_counts[i]}")


if __name__ == "__main__":
    verify_label_correction()