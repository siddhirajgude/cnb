import os
import yaml
import random
import logging
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def set_seed(seed: int = 42):
    """
    Sets random seed for reproducibility across python, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(name: str = "cnb_ai_logger") -> logging.Logger:
    """Configures standard logging output for Core Needle Biopsy AI pipeline."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def save_checkpoint(state: dict, is_best: bool = False, save_dir: str = "models", filename: str = "checkpoint.pt"):
    """Saves model checkpoint for automatic recovery."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Loads model checkpoint and restores weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> dict:
    """
    Calculates PRD Section 9 Evaluation Metrics:
    Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    auc = None
    if y_probs is not None:
        try:
            if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except Exception:
            auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1-Score': float(f1),
        'ROC-AUC': float(auc) if auc is not None else 0.5,
        'Confusion_Matrix': cm.tolist()
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: list = None, save_path: str = None):
    """Plots and saves confusion matrix heatmap."""
    if class_names is None:
        class_names = ['Benign', 'Malignant']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str = None):
    """Plots and saves ROC Curve."""
    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, save_path: str = None):
    """Plots training & validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metrics_comparison(df, metric_col: str, title: str, ylabel: str, save_path: str = None):
    """Generates bar plot comparing performance across all 40 noise-filter combinations."""
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df, x='Noise Type', y=metric_col, hue='Denoising Method', palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def save_image(img_array: np.ndarray, save_path: str):
    """Saves float image array [0, 1] as uint8 image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_uint8 = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_uint8)
