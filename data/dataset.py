import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List
import re


class SCUTDataLoader:


    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (80, 80)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        # Verify directory structure
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"

        if not self.train_dir.exists():
            raise ValueError(f"Training directory not found: {self.train_dir}")
        if not self.val_dir.exists():
            raise ValueError(f"Validation directory not found: {self.val_dir}")
        if not self.test_dir.exists():
            raise ValueError(f"Test directory not found: {self.test_dir}")

    def parse_filename(self, filepath: str) -> Tuple[str, int]:
        filename = os.path.basename(filepath)
        # Extract the attractiveness level from filename: <level>_<id>.jpg
        match = re.match(r"^(\d+)_.*\.jpg$", filename)

        if not match:
            raise ValueError(f"Invalid filename format: {filename}")

        label = int(match.group(1))
        return filepath, label

    def load_image_and_label(
        self, filepath: str, label: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Read image file
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize(img, self.img_size)

        # Normalize pixel values to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0

        # Convert label to float for regression
        label = tf.cast(label, tf.float32)

        return img, label

    def get_dataset(
        self,
        split: str,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
    ) -> tf.data.Dataset:
        if split == "train":
            data_dir = self.train_dir
        elif split == "val" or split == "validation":
            data_dir = self.val_dir
        elif split == "test":
            data_dir = self.test_dir
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
            )

        image_files = sorted([str(f) for f in data_dir.glob("*.jpg")])

        if len(image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(image_files)} images in {split} set")

        filepaths = []
        labels = []
        for filepath in image_files:
            fp, label = self.parse_filename(filepath)
            filepaths.append(fp)
            labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(filepaths))

        dataset = dataset.map(
            lambda fp, lbl: tf.py_function(
                func=self.load_image_and_label,
                inp=[fp, lbl],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.map(
            lambda img, lbl: (
                tf.ensure_shape(img, [*self.img_size, 3]),
                tf.ensure_shape(lbl, []),
            )
        )

        if augment and split == "train":
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


    #opt
    def _augment(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label


