"""
Quick script to verify the original dataset structure
Save this as verify_dataset.py and run it separately
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def verify_original_dataset():
    """Verify the original dataset files"""

    # Adjust this path to your dataset location
    dataset_path = r"D:\Software enginner\university\sem7\GenAI\ASS-01\data\raw\sign-language-digits-dataset"

    x_file = os.path.join(dataset_path, 'X.npy')
    y_file = os.path.join(dataset_path, 'Y.npy')

    # Try lowercase if uppercase doesn't exist
    if not os.path.exists(x_file):
        x_file = os.path.join(dataset_path, 'x.npy')
        y_file = os.path.join(dataset_path, 'y.npy')

    if not os.path.exists(x_file) or not os.path.exists(y_file):
        print("Dataset files not found!")
        return

    print("Loading original dataset files...")
    X = np.load(x_file)
    Y = np.load(y_file)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"Y dtype: {Y.dtype}")
    print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Y unique values: {np.unique(Y)}")

    # Check if Y is one-hot encoded
    if len(Y.shape) == 2 and Y.shape[1] == 10:
        print("\nY appears to be one-hot encoded")
        print("First 10 one-hot vectors:")
        for i in range(10):
            one_hot = Y[i]
            class_idx = np.argmax(one_hot)
            print(f"  Sample {i}: {one_hot} -> Class {class_idx}")

        # Convert to class indices
        class_labels = np.argmax(Y, axis=1)
        print(f"\nConverted class distribution: {np.bincount(class_labels)}")

        # Show a few sample images with their labels
        fig, axes = plt.subplots(2, 10, figsize=(20, 8))
        fig.suptitle('Original Dataset - First 10 samples from each class')

        for digit in range(10):
            # Find first two samples of this digit
            digit_indices = np.where(class_labels == digit)[0]

            if len(digit_indices) > 0:
                # First sample
                idx = digit_indices[0]
                img = X[idx]

                # Handle different image formats
                if len(img.shape) == 2:  # Grayscale
                    axes[0, digit].imshow(img, cmap='gray')
                else:
                    axes[0, digit].imshow(img)

                axes[0, digit].set_title(f'Digit {digit}\nIdx: {idx}')
                axes[0, digit].axis('off')

                # Second sample if available
                if len(digit_indices) > 1:
                    idx2 = digit_indices[1]
                    img2 = X[idx2]

                    if len(img2.shape) == 2:
                        axes[1, digit].imshow(img2, cmap='gray')
                    else:
                        axes[1, digit].imshow(img2)

                    axes[1, digit].set_title(f'Digit {digit}\nIdx: {idx2}')
                    axes[1, digit].axis('off')
                else:
                    axes[1, digit].axis('off')

        plt.tight_layout()
        plt.show()

        # Check for potential ordering issues
        print("\nChecking for ordering patterns...")

        # Look at the first occurrence of each class
        first_occurrences = {}
        for i, label in enumerate(class_labels):
            if label not in first_occurrences:
                first_occurrences[label] = i

        print("First occurrence of each digit:")
        for digit in sorted(first_occurrences.keys()):
            print(f"  Digit {digit}: Index {first_occurrences[digit]}")

        # Check if data is sorted by class (which might cause issues)
        is_sorted = all(class_labels[i] <= class_labels[i + 1] for i in range(len(class_labels) - 1))
        print(f"Data is sorted by class: {is_sorted}")

        if is_sorted:
            print("WARNING: Data appears to be sorted by class. This might cause visualization issues.")
            print("Consider shuffling the data while maintaining image-label correspondence.")

    else:
        print("Y is not one-hot encoded or has unexpected format")


if __name__ == "__main__":
    verify_original_dataset()