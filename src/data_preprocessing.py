"""
Data preprocessing module for Sign Language Digits Recognition
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

    def load_numpy_dataset(self, dataset_path=None):
        """
        Load the sign language digits dataset from .npy files
        If dataset_path is None, assumes data is in RAW_DATA_DIR
        """
        if dataset_path is None:
            dataset_path = RAW_DATA_DIR

        # Check for the dataset folder structure
        sign_lang_folder = os.path.join(dataset_path, 'sign-language-digits-dataset')

        # Try different possible paths
        possible_paths = [
            sign_lang_folder,  # sign-language-digits-dataset folder
            dataset_path,      # directly in raw data directory
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
            print("Dataset .npy files not found. Please check the following:")
            print("1. Download the dataset from: https://www.kaggle.com/ardamavi/sign-language-digits-dataset")
            print("2. Extract the zip file")
            print("3. Place the 'sign-language-digits-dataset' folder in the data/raw/ directory")
            print("4. The folder should contain X.npy and Y.npy (or x.npy and y.npy) files")
            print(f"Expected paths checked:")
            for path in possible_paths:
                print(f"  - {os.path.join(path, 'X.npy')} or {os.path.join(path, 'x.npy')}")
            return False

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
            print(f"Image data type: {images.dtype}")
            print(f"Labels data type: {labels.dtype}")
            print(f"Image value range: [{images.min()}, {images.max()}]")
            print(f"Unique labels: {np.unique(labels)}")

            # Handle different possible image formats
            if len(images.shape) == 4:
                # Already in the correct format (N, H, W, C) or (N, H, W, 1)
                if images.shape[-1] == 1:
                    # Convert grayscale to RGB
                    images = np.repeat(images, 3, axis=-1)
                elif images.shape[-1] == 3:
                    # Already RGB
                    pass
                else:
                    print(f"Unexpected number of channels: {images.shape[-1]}")
                    return False
            elif len(images.shape) == 3:
                # Might be (N, H, W) - add channel dimension and convert to RGB
                images = np.expand_dims(images, axis=-1)
                images = np.repeat(images, 3, axis=-1)
            else:
                print(f"Unexpected image shape: {images.shape}")
                return False

            # Resize images if they're not 64x64
            if images.shape[1:3] != (64, 64):
                print(f"Resizing images from {images.shape[1:3]} to (64, 64)")
                resized_images = []
                for i, img in enumerate(images):
                    if i % 1000 == 0:
                        print(f"Resizing image {i}/{len(images)}")

                    # Handle different data types
                    if img.dtype != np.uint8:
                        # Normalize to 0-255 if not already
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)

                    # Resize image
                    if img.shape[2] == 1:  # Grayscale
                        resized = cv2.resize(img[:, :, 0], (64, 64))
                        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    else:  # RGB
                        resized = cv2.resize(img, (64, 64))
                        if img.shape[2] == 3:
                            # Convert BGR to RGB if needed (OpenCV uses BGR)
                            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    resized_images.append(resized)

                images = np.array(resized_images)
            else:
                # Images are already 64x64, just ensure they're in the right format
                if images.dtype != np.uint8:
                    if images.max() <= 1.0:
                        images = (images * 255).astype(np.uint8)
                    else:
                        images = images.astype(np.uint8)

            # Ensure labels are in the correct format
            if labels.ndim > 1:
                # If labels are one-hot encoded, convert back to class indices
                if labels.shape[1] == NUM_CLASSES:
                    labels = np.argmax(labels, axis=1)

            # Validate data
            assert len(images) == len(labels), f"Mismatch between images ({len(images)}) and labels ({len(labels)})"
            assert images.shape[1:] == (64, 64, 3), f"Images should be (64, 64, 3), got {images.shape[1:]}"
            assert np.all(labels >= 0) and np.all(labels < NUM_CLASSES), f"Invalid label range: {labels.min()} to {labels.max()}"

            self.image_data = images
            self.labels = labels

            print(f"Dataset loaded successfully!")
            print(f"Final image shape: {self.image_data.shape}")
            print(f"Final labels shape: {self.labels.shape}")
            print(f"Image value range: [{self.image_data.min()}, {self.image_data.max()}]")
            print(f"Label range: [{self.labels.min()}, {self.labels.max()}]")

            return True

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_kaggle_dataset(self, dataset_path=None):
        """
        Load the sign language digits dataset - Updated to use numpy files
        This method now calls load_numpy_dataset for backward compatibility
        """
        return self.load_numpy_dataset(dataset_path)

    def load_image_directory(self, dataset_path):
        """
        Alternative method to load from image directories
        Expects structure: dataset_path/0/, dataset_path/1/, ..., dataset_path/9/
        """
        images = []
        labels = []

        for digit in range(10):
            digit_path = os.path.join(dataset_path, str(digit))
            if os.path.exists(digit_path):
                for filename in os.listdir(digit_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(digit_path, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize to 64x64
                            img = cv2.resize(img, IMAGE_SIZE)
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(digit)

        if len(images) > 0:
            self.image_data = np.array(images)
            self.labels = np.array(labels)
            return True
        else:
            return False

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
        Visualize sample images from each class
        """
        if self.image_data is None or self.labels is None:
            raise ValueError("No data loaded. Call load_dataset first.")

        fig, axes = plt.subplots(2, 10, figsize=(15, 6))
        fig.suptitle('Sample Sign Language Digit Images', fontsize=16)

        # Get one sample from each class
        for digit in range(10):
            # Find indices of current digit
            digit_indices = np.where(self.labels == digit)[0]
            if len(digit_indices) > 0:
                # Select first sample
                sample_idx = digit_indices[0]
                img = self.image_data[sample_idx]

                # Ensure image is in displayable format
                if img.dtype == np.float32:
                    # If normalized, scale back for display
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)

                # Display in first row
                axes[0, digit].imshow(img)
                axes[0, digit].set_title(f'Digit {digit}')
                axes[0, digit].axis('off')

                # Display another sample in second row if available
                if len(digit_indices) > 1:
                    sample_idx = digit_indices[1]
                    img = self.image_data[sample_idx]

                    if img.dtype == np.float32:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)

                    axes[1, digit].imshow(img)
                    axes[1, digit].axis('off')
                else:
                    axes[1, digit].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

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

        def list_directory_contents(path, max_depth=2, current_depth=0):
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
        print("data/raw/sign-language-digits-dataset/")
        print("  X.npy (or x.npy) - Image data")
        print("  Y.npy (or y.npy) - Label data")
        print("\nIf you have a different structure, please reorganize accordingly.")