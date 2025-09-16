"""
Data preprocessing module for Sign Language Digits Recognition - FIXED FOR FOLDER STRUCTURE
"""
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from config import *


class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.image_data = None
        self.labels = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_folder_dataset(self, dataset_path=None):
        """
        Load the sign language digits dataset from folder structure (0/, 1/, 2/, ..., 9/)
        """
        if dataset_path is None:
            dataset_path = RAW_DATA_DIR

        # Check for the dataset folder structure
        possible_paths = [
            os.path.join(dataset_path, 'Dataset'),  # Dataset folder from GitHub
            os.path.join(dataset_path, 'sign-language-digits-dataset'),  # Alternative naming
            dataset_path,  # directly in raw data directory
        ]

        dataset_found = False
        actual_path = None

        # Find the correct dataset path
        for path in possible_paths:
            # Check if digit folders exist (0, 1, 2, ..., 9)
            digit_folders_exist = all(
                os.path.exists(os.path.join(path, str(digit)))
                for digit in range(10)
            )

            if digit_folders_exist:
                dataset_found = True
                actual_path = path
                print(f"Found dataset structure at: {actual_path}")
                break

        if not dataset_found:
            print("Dataset folder structure not found. Please check the following:")
            print("1. Download the dataset from: https://github.com/ardamavi/Sign-Language-Digits-Dataset")
            print("2. Extract and place the 'Dataset' folder in the data/raw/ directory")
            print("3. The folder should contain subfolders: 0/, 1/, 2/, ..., 9/")
            print("4. Each subfolder should contain image files (.png, .jpg, .jpeg)")
            return False

        try:
            print(f"Loading dataset from folder structure: {actual_path}")

            images = []
            labels = []

            # Load images from each digit folder
            for digit in range(10):
                digit_folder = os.path.join(actual_path, str(digit))
                print(f"Loading digit {digit} from: {digit_folder}")

                if not os.path.exists(digit_folder):
                    print(f"Warning: Folder for digit {digit} not found")
                    continue

                # Get all image files in the digit folder
                image_files = [f for f in os.listdir(digit_folder)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                print(f"  Found {len(image_files)} images for digit {digit}")

                for img_file in image_files:
                    img_path = os.path.join(digit_folder, img_file)

                    # Load image
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Resize to target size (64x64)
                        img = cv2.resize(img, IMAGE_SIZE)

                        images.append(img)
                        labels.append(digit)
                    else:
                        print(f"  Warning: Could not load image {img_path}")

            if len(images) == 0:
                print("No images were loaded. Please check the dataset structure.")
                return False

            # Convert to numpy arrays
            self.image_data = np.array(images, dtype=np.uint8)
            self.labels = np.array(labels, dtype=np.int32)

            print(f"Dataset loaded successfully!")
            print(f"Total images loaded: {len(self.image_data)}")
            print(f"Image shape: {self.image_data.shape}")
            print(f"Labels shape: {self.labels.shape}")
            print(f"Image data type: {self.image_data.dtype}")
            print(f"Labels data type: {self.labels.dtype}")
            print(f"Image value range: [{self.image_data.min()}, {self.image_data.max()}]")
            print(f"Label range: [{self.labels.min()}, {self.labels.max()}]")

            # Show class distribution
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print("Class distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  Digit {label}: {count} samples")

            # Verify data integrity
            print("\nVerifying data integrity...")
            assert len(self.image_data) == len(self.labels), f"Data length mismatch: {len(self.image_data)} images vs {len(self.labels)} labels"
            assert self.image_data.shape[1:] == (64, 64, 3), f"Images should be (64, 64, 3), got {self.image_data.shape[1:]}"
            assert np.all(self.labels >= 0) and np.all(self.labels < NUM_CLASSES), f"Invalid label range: {self.labels.min()} to {self.labels.max()}"
            assert len(unique_labels) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, found {len(unique_labels)}"

            print("Data integrity check passed!")
            return True

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_numpy_dataset(self, dataset_path=None):
        """
        Load the sign language digits dataset from .npy files - LEGACY METHOD
        """
        print("Attempting to load from .npy files...")

        if dataset_path is None:
            dataset_path = RAW_DATA_DIR

        # Check for the dataset folder structure
        sign_lang_folder = os.path.join(dataset_path, 'sign-language-digits-dataset')

        # Try different possible paths
        possible_paths = [
            sign_lang_folder,  # sign-language-digits-dataset folder
            dataset_path,  # directly in raw data directory
            os.path.join(dataset_path, 'data')  # in case there's a data subfolder
        ]

        dataset_found = False
        actual_path = None

        for path in possible_paths:
            x_file = os.path.join(path, 'X.npy')  # Try uppercase first
            y_file = os.path.join(path, 'Y.npy')

            # If uppercase doesn't exist, try lowercase
            if not os.path.exists(x_file):
                x_file = os.path.join(path, 'x.npy')
                y_file = os.path.join(path, 'y.npy')

            if os.path.exists(x_file) and os.path.exists(y_file):
                dataset_found = True
                actual_path = path
                break

        if not dataset_found:
            print(".npy files not found. Trying folder structure instead...")
            return self.load_folder_dataset(dataset_path)

        try:
            # Load the numpy arrays
            print(f"Loading dataset from: {actual_path}")

            # Try uppercase first, then lowercase
            x_file = os.path.join(actual_path, 'X.npy')
            y_file = os.path.join(actual_path, 'Y.npy')

            if not os.path.exists(x_file):
                x_file = os.path.join(actual_path, 'x.npy')
                y_file = os.path.join(actual_path, 'y.npy')

            images = np.load(x_file)
            labels = np.load(y_file)

            print(f"Loaded images shape: {images.shape}")
            print(f"Loaded labels shape: {labels.shape}")

            # Handle one-hot encoded labels if necessary
            if len(labels.shape) == 2 and labels.shape[1] == NUM_CLASSES:
                print("Labels are one-hot encoded. Converting to class indices...")
                labels = np.argmax(labels, axis=1)

            # Handle different image formats and resize if needed
            if len(images.shape) == 4:
                if images.shape[-1] == 1:
                    # Convert grayscale to RGB
                    images = np.repeat(images, 3, axis=-1)
                elif images.shape[-1] == 3:
                    pass  # Already RGB
            elif len(images.shape) == 3:
                # Add channel dimension and convert to RGB
                images = np.expand_dims(images, axis=-1)
                images = np.repeat(images, 3, axis=-1)

            # Resize if needed
            if images.shape[1:3] != (64, 64):
                print(f"Resizing images from {images.shape[1:3]} to (64, 64)")
                resized_images = []
                for img in images:
                    resized = cv2.resize(img, (64, 64))
                    if len(resized.shape) == 2:
                        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    resized_images.append(resized)
                images = np.array(resized_images)

            # Ensure proper data types
            if images.dtype != np.uint8:
                if images.max() <= 1.0:
                    images = (images * 255).astype(np.uint8)
                else:
                    images = images.astype(np.uint8)

            self.image_data = images
            self.labels = labels.astype(np.int32)

            print(f"Dataset loaded successfully from .npy files!")
            return True

        except Exception as e:
            print(f"Error loading .npy files: {str(e)}")
            print("Falling back to folder structure...")
            return self.load_folder_dataset(dataset_path)

    def load_kaggle_dataset(self, dataset_path=None):
        """
        Load the sign language digits dataset - Updated to try folder structure first
        """
        print("Loading Sign Language Digits dataset...")

        # First try folder structure (which is the correct format for the GitHub dataset)
        if self.load_folder_dataset(dataset_path):
            return True

        # Fallback to .npy files if folder structure doesn't work
        print("Folder structure failed, trying .npy files...")
        return self.load_numpy_dataset(dataset_path)

    def load_image_directory(self, dataset_path):
        """
        Alternative method to load from image directories
        Expects structure: dataset_path/0/, dataset_path/1/, ..., dataset_path/9/
        """
        return self.load_folder_dataset(dataset_path)

    def normalize_data(self, normalization_type='standard'):
        """
        Normalize the image data
        """
        if self.image_data is None:
            raise ValueError("No data loaded. Call load_dataset first.")

        if normalization_type == 'standard':
            # Normalize to [0, 1] range
            self.image_data = self.image_data.astype('float32') / 255.0
        elif normalization_type == 'zscore':
            # Z-score normalization
            mean = np.mean(self.image_data, axis=(0, 1, 2))
            std = np.std(self.image_data, axis=(0, 1, 2))
            self.image_data = (self.image_data - mean) / (std + 1e-8)
        elif normalization_type == 'minmax':
            # Min-max normalization
            min_val = np.min(self.image_data)
            max_val = np.max(self.image_data)
            self.image_data = (self.image_data - min_val) / (max_val - min_val)

    def split_data(self, test_size=0.1, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets
        """
        if self.image_data is None or self.labels is None:
            raise ValueError("No data loaded. Call load_dataset first.")

        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.image_data, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )

        # Convert labels to categorical
        self.y_train = to_categorical(self.y_train, NUM_CLASSES)
        self.y_val = to_categorical(self.y_val, NUM_CLASSES)
        self.y_test = to_categorical(self.y_test, NUM_CLASSES)

        print(f"Data split completed:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

    def create_data_augmentation(self):
        """
        Create data augmentation generator
        """
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip sign language gestures
            fill_mode='nearest'
        )
        return datagen

    def visualize_samples(self, num_samples=20, save_path=None):
        """
        Visualize sample images from each class - CORRECTED VERSION
        """
        if self.image_data is None or self.labels is None:
            raise ValueError("No data loaded. Call load_dataset first.")

        fig, axes = plt.subplots(2, 10, figsize=(15, 6))
        fig.suptitle('Sample Sign Language Digit Images', fontsize=16)

        print("\n=== VERIFYING IMAGE-LABEL ALIGNMENT ===")

        # Verify data integrity first
        for digit in range(10):
            # Find indices of current digit
            digit_indices = np.where(self.labels == digit)[0]

            print(f"\nDigit {digit}:")
            print(f"  Found {len(digit_indices)} samples")

            if len(digit_indices) > 0:
                # Get first sample
                sample_idx = digit_indices[0]
                img = self.image_data[sample_idx]
                actual_label = self.labels[sample_idx]

                print(f"  Sample index: {sample_idx}")
                print(f"  Actual label: {actual_label}")
                print(f"  Expected label: {digit}")

                # Verification: Check if labels match
                if actual_label != digit:
                    print(f"  ⚠️  MISMATCH DETECTED! Expected {digit}, got {actual_label}")
                else:
                    print(f"  ✓ Labels match correctly")

                # Prepare image for display
                display_img = img.copy()
                if display_img.dtype == np.float32:
                    if display_img.max() <= 1.0:
                        display_img = (display_img * 255).astype(np.uint8)

                # Display first sample
                if len(display_img.shape) == 3 and display_img.shape[2] == 3:
                    axes[0, digit].imshow(display_img)
                else:
                    axes[0, digit].imshow(display_img, cmap='gray')

                axes[0, digit].set_title(f'Digit {digit}\n(Label: {actual_label})')
                axes[0, digit].axis('off')

                # Display second sample if available
                if len(digit_indices) > 1:
                    sample_idx2 = digit_indices[1]
                    img2 = self.image_data[sample_idx2]
                    actual_label2 = self.labels[sample_idx2]

                    display_img2 = img2.copy()
                    if display_img2.dtype == np.float32:
                        if display_img2.max() <= 1.0:
                            display_img2 = (display_img2 * 255).astype(np.uint8)

                    if len(display_img2.shape) == 3 and display_img2.shape[2] == 3:
                        axes[1, digit].imshow(display_img2)
                    else:
                        axes[1, digit].imshow(display_img2, cmap='gray')

                    axes[1, digit].set_title(f'Digit {digit}\n(Label: {actual_label2})')
                    axes[1, digit].axis('off')
                else:
                    axes[1, digit].axis('off')
            else:
                print(f"  ⚠️  No samples found for digit {digit}")
                axes[0, digit].text(0.5, 0.5, f'No samples\nfor digit {digit}',
                                    ha='center', va='center', transform=axes[0, digit].transAxes)
                axes[0, digit].axis('off')
                axes[1, digit].axis('off')

        print("\n=== END VERIFICATION ===")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Additional verification: Check overall label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"\nLabel distribution verification:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} samples")

    def get_class_distribution(self):
        """
        Get class distribution statistics
        """
        if self.labels is None:
            raise ValueError("No data loaded. Call load_dataset first.")

        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        # Visualize distribution
        plt.figure(figsize=(10, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.xlabel('Digit Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Dataset')
        plt.xticks(range(10))

        for i, count in enumerate(counts):
            plt.text(i, count + 10, str(count), ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=300)
        plt.show()

        return distribution

    def save_processed_data(self):
        """
        Save processed data to disk
        """
        if self.X_train is None:
            raise ValueError("Data not split yet. Call split_data first.")

        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), self.X_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), self.X_val)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), self.X_test)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), self.y_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), self.y_val)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), self.y_test)

        print("Processed data saved to:", PROCESSED_DATA_DIR)

    def load_processed_data(self):
        """
        Load previously processed data
        """
        try:
            self.X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
            self.X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
            self.X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
            self.y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
            self.y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
            self.y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))

            print("Processed data loaded successfully")
            return True
        except FileNotFoundError:
            print("Processed data files not found. Need to process data first.")
            return False

    def inspect_dataset_structure(self, dataset_path=None):
        """
        Helper method to inspect the dataset structure and provide guidance
        """
        if dataset_path is None:
            dataset_path = RAW_DATA_DIR

        print(f"Inspecting dataset structure in: {dataset_path}")
        print("=" * 50)

        def list_directory_contents(path, max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return

            indent = "  " * current_depth

            if os.path.exists(path):
                print(f"{indent}{os.path.basename(path)}/")
                try:
                    items = os.listdir(path)
                    for item in sorted(items):
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            list_directory_contents(item_path, max_depth, current_depth + 1)
                        else:
                            size = os.path.getsize(item_path)
                            print(f"{indent}  {item} ({size:,} bytes)")
                except PermissionError:
                    print(f"{indent}  [Permission denied]")
            else:
                print(f"{indent}[Path does not exist: {path}]")

        list_directory_contents(dataset_path)

        print("\n" + "=" * 50)
        print("Expected structure for this dataset:")
        print("data/raw/Dataset/")
        print("  0/ - Images for digit 0")
        print("  1/ - Images for digit 1")
        print("  2/ - Images for digit 2")
        print("  ...")
        print("  9/ - Images for digit 9")
        print("\nAlternatively:")
        print("data/raw/sign-language-digits-dataset/")
        print("  X.npy - Image data")
        print("  Y.npy - Label data")

    def verify_data_alignment(self):
        """
        Comprehensive verification of image-label alignment
        """
        if self.image_data is None or self.labels is None:
            print("No data loaded for verification")
            return False

        print("=== COMPREHENSIVE DATA VERIFICATION ===")

        # Basic checks
        print(f"Image data shape: {self.image_data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Image dtype: {self.image_data.dtype}")
        print(f"Labels dtype: {self.labels.dtype}")

        # Length check
        if len(self.image_data) != len(self.labels):
            print(f"❌ LENGTH MISMATCH: {len(self.image_data)} images vs {len(self.labels)} labels")
            return False

        # Label range check
        unique_labels = np.unique(self.labels)
        expected_labels = np.arange(10)

        print(f"Unique labels found: {unique_labels}")
        print(f"Expected labels: {expected_labels}")

        if not np.array_equal(unique_labels, expected_labels):
            print("❌ LABEL RANGE ISSUE: Labels are not 0-9 consecutive")
            return False

        # Class distribution check
        print("\n--- Class Distribution ---")
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Digit {label}: {count} samples ({count / len(self.labels) * 100:.1f}%)")

        # Check for extreme imbalances
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > 3.0:
            print(f"⚠️  CLASS IMBALANCE WARNING: Ratio {imbalance_ratio:.2f}")
        else:
            print(f"✓ Class balance is reasonable (ratio: {imbalance_ratio:.2f})")

        print("=== END VERIFICATION ===")
        return True