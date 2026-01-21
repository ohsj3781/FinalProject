"""
COCO Dataset Loader for Multi-Label Classification

This module provides a PyTorch Dataset implementation for loading COCO 2017 dataset
in a multi-label classification format (80 classes).
"""

import os
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


class COCOMultiLabelDataset(Dataset):
    """
    COCO Dataset for Multi-Label Classification.

    Each image can have multiple labels corresponding to the objects present in it.
    Labels are represented as a binary vector of size 80 (number of COCO categories).

    Args:
        root: Root directory of COCO dataset
        annFile: Path to annotation JSON file
        transform: Optional transform to be applied on images
        target_transform: Optional transform to be applied on labels
    """

    # COCO category IDs are not contiguous (1-90 with gaps)
    # Map them to contiguous indices (0-79)
    COCO_CAT_IDS = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90
    ]

    NUM_CLASSES = 80

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # Load COCO annotations
        self.coco = COCO(annFile)

        # Get all image IDs
        self.img_ids = list(self.coco.imgs.keys())

        # Create category ID to index mapping
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.COCO_CAT_IDS)}

        # Get category names
        self.categories = self.coco.loadCats(self.COCO_CAT_IDS)
        self.cat_names = [cat['name'] for cat in self.categories]

        # Pre-compute labels for all images
        self._precompute_labels()

    def _precompute_labels(self):
        """Pre-compute multi-label vectors for all images."""
        self.labels = {}

        for img_id in self.img_ids:
            # Get all annotation IDs for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Create binary label vector
            label = np.zeros(self.NUM_CLASSES, dtype=np.float32)

            for ann in anns:
                cat_id = ann['category_id']
                if cat_id in self.cat_id_to_idx:
                    idx = self.cat_id_to_idx[cat_id]
                    label[idx] = 1.0

            self.labels[img_id] = label

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and multi-label target.

        Args:
            index: Index of the sample

        Returns:
            tuple: (image, target) where target is a binary vector of shape (80,)
        """
        img_id = self.img_ids[index]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Get pre-computed label
        target = torch.from_numpy(self.labels[img_id])

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def get_image_info(self, index: int) -> Dict:
        """Get metadata for an image."""
        img_id = self.img_ids[index]
        return self.coco.loadImgs(img_id)[0]

    def get_labels_for_image(self, index: int) -> List[str]:
        """Get list of category names present in an image."""
        img_id = self.img_ids[index]
        label = self.labels[img_id]
        return [self.cat_names[i] for i in range(self.NUM_CLASSES) if label[i] == 1.0]

    def get_category_names(self) -> List[str]:
        """Get list of all category names."""
        return self.cat_names

    def get_class_frequencies(self) -> np.ndarray:
        """
        Compute frequency of each class across the dataset.
        Useful for computing class weights for imbalanced data.
        """
        frequencies = np.zeros(self.NUM_CLASSES, dtype=np.float32)

        for label in self.labels.values():
            frequencies += label

        return frequencies / len(self.labels)

    def get_pos_weights(self) -> torch.Tensor:
        """
        Compute positive class weights for BCEWithLogitsLoss.
        weight = (num_negative) / (num_positive) for each class
        """
        total = len(self.labels)
        frequencies = self.get_class_frequencies()

        # Avoid division by zero
        frequencies = np.clip(frequencies, 1e-6, 1.0 - 1e-6)

        # pos_weight = num_neg / num_pos = (1 - freq) / freq
        pos_weights = (1.0 - frequencies) / frequencies

        return torch.from_numpy(pos_weights)


def create_coco_dataloaders(
    data_dir: str,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for COCO dataset.

    Args:
        data_dir: Root directory containing COCO dataset
        train_transform: Transform for training images
        val_transform: Transform for validation images
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Paths
    train_img_dir = os.path.join(data_dir, 'train2017')
    val_img_dir = os.path.join(data_dir, 'val2017')
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    # Create datasets
    train_dataset = COCOMultiLabelDataset(
        root=train_img_dir,
        annFile=train_ann_file,
        transform=train_transform
    )

    val_dataset = COCOMultiLabelDataset(
        root=val_img_dir,
        annFile=val_ann_file,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Example usage (update paths as needed)
    data_dir = "data/coco"

    if os.path.exists(data_dir):
        train_loader, val_loader = create_coco_dataloaders(
            data_dir=data_dir,
            train_transform=transform,
            val_transform=transform,
            batch_size=32
        )

        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")

        # Get a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels per image: {labels.sum(dim=1).mean():.2f} (average)")
