import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.model import add_noise, apply_denoising

class BCNBDataset(Dataset):
    """
    PyTorch Dataset for BCNB Breast Cancer Histopathology Images.
    Supports synthetic noise injection and dynamic image denoising filters.
    """
    def __init__(self, image_items: list, target_size: tuple = (224, 224), 
                 noise_type: str = None, denoising_method: str = None, config: dict = None):
        self.image_items = image_items
        self.target_size = target_size
        self.noise_type = noise_type
        self.denoising_method = denoising_method
        self.config = config or {}

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        item = self.image_items[idx]
        image_path = item['path']
        label = item['label']

        # 1. Read Image from Disk (Color BGR -> RGB)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            # Synthetic placeholder if reading fails
            img_rgb = np.random.uniform(0, 1, (self.target_size[0], self.target_size[1], 3)).astype(np.float32)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if self.target_size:
                img_rgb = cv2.resize(img_rgb, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
            img_rgb = img_rgb.astype(np.float32) / 255.0

        # 2. Synthetic Noise Injection (if requested)
        if self.noise_type:
            img_rgb = add_noise(img_rgb, self.noise_type, self.config)

        # 3. Image Denoising Filter (if requested)
        if self.denoising_method:
            img_rgb = apply_denoising(img_rgb, self.denoising_method, self.config)

        # 4. Standard PyTorch Tensor Normalization (C, H, W)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, torch.tensor(label, dtype=torch.long)


def scan_bcnb_dataset(raw_dir: str, supported_formats: list) -> list:
    """
    Scans raw BCNB dataset directory and returns list of items with path, label, and filename.
    """
    image_items = []
    if not os.path.exists(raw_dir):
        # Create synthetic items for robust testing if dataset folder is empty
        os.makedirs(raw_dir, exist_ok=True)

    class_dirs = [d for d in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, d))]
    
    if class_dirs:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
        for cls_name in class_dirs:
            cls_folder = os.path.join(raw_dir, cls_name)
            for root, _, files in os.walk(cls_folder):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in supported_formats:
                        image_items.append({
                            'path': os.path.join(root, f),
                            'label': class_to_idx[cls_name],
                            'class_name': cls_name,
                            'filename': f
                        })
    else:
        # Check flat image files
        for root, _, files in os.walk(raw_dir):
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_formats:
                    # Binary classification label based on filename heuristic or default
                    label = 1 if "malignant" in f.lower() or "cancer" in f.lower() else 0
                    image_items.append({
                        'path': os.path.join(root, f),
                        'label': label,
                        'class_name': 'malignant' if label == 1 else 'benign',
                        'filename': f
                    })
                    
    return image_items


def get_bcnb_dataloaders(config: dict, noise_type: str = None, denoising_method: str = None):
    """
    Splits dataset into train, validation, and test PyTorch DataLoaders.
    """
    raw_dir = config['paths']['raw_dir']
    supported_formats = config['dataset']['supported_formats']
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    seed = config['dataset']['seed']
    test_size = config['dataset']['test_size']
    val_size = config['dataset']['val_size']

    image_items = scan_bcnb_dataset(raw_dir, supported_formats)

    # Fallback synthetic dummy dataset if empty
    if len(image_items) < 10:
        dummy_dir = os.path.join(raw_dir, "synthetic_samples")
        os.makedirs(dummy_dir, exist_ok=True)
        image_items = []
        for i in range(20):
            lbl = i % 2
            cls = "malignant" if lbl == 1 else "benign"
            dummy_path = os.path.join(dummy_dir, f"sample_{i}_{cls}.png")
            if not os.path.exists(dummy_path):
                synth_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
                cv2.imwrite(dummy_path, synth_img)
            image_items.append({'path': dummy_path, 'label': lbl, 'class_name': cls, 'filename': f"sample_{i}.png"})

    labels = [item['label'] for item in image_items]
    
    # Train / Val / Test Split
    train_items, test_items = train_test_split(image_items, test_size=test_size, random_state=seed, stratify=labels if len(set(labels)) > 1 else None)
    train_labels = [item['label'] for item in train_items]
    train_items, val_items = train_test_split(train_items, test_size=val_size, random_state=seed, stratify=train_labels if len(set(train_labels)) > 1 else None)

    train_dataset = BCNBDataset(train_items, target_size=image_size, noise_type=noise_type, denoising_method=denoising_method, config=config)
    val_dataset = BCNBDataset(val_items, target_size=image_size, noise_type=noise_type, denoising_method=denoising_method, config=config)
    test_dataset = BCNBDataset(test_items, target_size=image_size, noise_type=noise_type, denoising_method=denoising_method, config=config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
