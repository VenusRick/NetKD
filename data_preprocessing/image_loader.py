"""
Image-based dataset loader for NetKD.

This module loads pre-processed traffic flow images directly from disk.
支持已经预处理好的流量图像数据集（PNG格式）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ImageFlowDataset(Dataset):
    """
    Dataset for loading pre-processed traffic flow images.
    
    支持的数据集结构:
    dataset_root/
        ├── images_sampled_new/
        │   ├── class1/
        │   │   ├── image1.png
        │   │   ├── image2.png
        │   │   └── ...
        │   ├── class2/
        │   │   └── ...
        │   └── ...
    
    Args:
        image_paths: List of image file paths
        labels: List of integer labels corresponding to images
        label_names: Optional list of class names
        transform: Optional transform to apply to images
        to_grayscale: Convert images to grayscale (default: True)
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        label_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        to_grayscale: bool = True,
    ):
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        self.image_paths = image_paths
        self.labels = labels
        self.label_names = label_names
        self.transform = transform
        self.to_grayscale = to_grayscale
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        # Convert to grayscale if needed
        if self.to_grayscale and img.mode != 'L':
            img = img.convert('L')
        elif not self.to_grayscale and img.mode == 'L':
            img = img.convert('RGB')
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor
        if img_array.ndim == 2:  # Grayscale
            x = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        else:  # RGB
            x = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        
        # Apply transform if provided
        if self.transform:
            x = self.transform(x)
        
        # Get label
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index."""
        if self.label_names is not None and 0 <= label < len(self.label_names):
            return self.label_names[label]
        return f"class_{label}"


def _detect_image_root(base_path: Path) -> Path:
    """Resolve the folder that actually holds class subdirectories of images.

    Historically, some datasets were stored under ``images_sampled_new`` while
    others exposed the class folders at the dataset root. This helper scans a
    few common layouts and returns the first one that clearly contains PNG
    files inside subdirectories.
    """

    candidates: Deque[Path] = deque(
        [
            base_path / "images_sampled_new",
            base_path / "images",
            base_path,
        ]
    )
    seen: set[Path] = set()

    while candidates:
        candidate = candidates.popleft()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)

        subdirs = [child for child in candidate.iterdir() if child.is_dir()]
        if not subdirs:
            continue

        for child in subdirs:
            if any(child.glob("*.png")):
                return candidate

        # Fall back to exploring one more level (train/val style layouts).
        candidates.extend(subdirs)

    raise FileNotFoundError(
        f"Could not locate image folders under {base_path}. Expected subdirectories with PNG files."
    )


def load_dataset_from_folders(
    dataset_root: str,
    dataset_name: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_seed: int = 42,
    to_grayscale: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load dataset from folder structure and create train/val/test dataloaders.
    
    Args:
        dataset_root: Root directory containing datasets (e.g., "G:\\数据集\\Dataset")
        dataset_name: Name of the dataset (e.g., "ISCXVPN2016", "ISCXTor2016", "USTC-TFC2016")
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        random_seed: Random seed for reproducibility
        to_grayscale: Convert images to grayscale
        
    Returns:
        train_loader, val_loader, test_loader, metadata_dict
    """
    
    # Construct dataset path
    base_path = Path(dataset_root) / dataset_name
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {base_path}")

    dataset_path = _detect_image_root(base_path)
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Scan for class folders
    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {dataset_path}")
    
    print(f"Found {len(class_folders)} classes: {[c.name for c in class_folders]}")
    
    # Build image paths and labels
    image_paths = []
    labels = []
    label_names = []
    
    for label_idx, class_folder in enumerate(class_folders):
        label_names.append(class_folder.name)
        
        # Find all PNG images in this class folder
        class_images = list(class_folder.glob("*.png"))
        
        print(f"  Class '{class_folder.name}': {len(class_images)} images")
        
        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(label_idx)
    
    print(f"\nTotal images loaded: {len(image_paths)}")
    print(f"Number of classes: {len(label_names)}")
    
    # Split into train/val/test
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_ratio, random_state=random_seed, stratify=labels
    )
    
    # Second split: train vs val
    val_size = val_ratio / (1 - test_ratio)  # Adjust val_ratio for remaining data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size, random_state=random_seed, stratify=train_val_labels
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    # Create datasets
    train_dataset = ImageFlowDataset(train_paths, train_labels, label_names, to_grayscale=to_grayscale)
    val_dataset = ImageFlowDataset(val_paths, val_labels, label_names, to_grayscale=to_grayscale)
    test_dataset = ImageFlowDataset(test_paths, test_labels, label_names, to_grayscale=to_grayscale)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Get image dimensions from first image
    sample_img = Image.open(image_paths[0])
    if to_grayscale and sample_img.mode != 'L':
        sample_img = sample_img.convert('L')
    img_array = np.array(sample_img)
    height, width = img_array.shape[:2]
    channels = 1 if to_grayscale else (3 if img_array.ndim == 3 else 1)
    
    # Metadata
    metadata = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "num_classes": len(label_names),
        "class_names": label_names,
        "total_images": len(image_paths),
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "test_size": len(test_paths),
        "image_height": height,
        "image_width": width,
        "image_channels": channels,
        "batch_size": batch_size,
    }
    
    print(f"\nImage dimensions: {channels} x {height} x {width}")
    
    return train_loader, val_loader, test_loader, metadata


def quick_load_dataset(
    dataset_name: str = "ISCXVPN2016",
    dataset_root: str | Path = "/walnut_data/yqm/Dataset",
    batch_size: int = 32,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Quick wrapper for loading common datasets.
    
    Supported datasets:
    - ISCXVPN2016: VPN traffic classification (7 classes)
    - ISCXTor2016: Tor traffic classification (8 classes)
    - USTC-TFC2016: Malware & application traffic (18 classes)
    - CrossPlatform-Android: Android app traffic
    - CrossPlatform-iOS: iOS app traffic
    - CICIoT2022: IoT device traffic
    
    Example:
        >>> train_loader, val_loader, test_loader, info = quick_load_dataset("ISCXVPN2016")
        >>> print(f"Loaded {info['num_classes']} classes")
    """
    return load_dataset_from_folders(
        dataset_root=str(dataset_root),
        dataset_name=dataset_name,
        batch_size=batch_size,
        **kwargs
    )


__all__ = [
    "ImageFlowDataset",
    "load_dataset_from_folders",
    "quick_load_dataset",
]
