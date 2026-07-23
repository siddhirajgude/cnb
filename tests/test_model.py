import os
import torch
import pytest
import numpy as np
from src.model import (
    CoreNeedleBiopsyClassifier,
    add_noise,
    apply_denoising
)
from src.utils import (
    calculate_classification_metrics,
    save_checkpoint,
    load_checkpoint,
    set_seed
)

@pytest.fixture
def sample_image():
    set_seed(42)
    return np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)

@pytest.fixture
def config():
    return {
        'noise': {
            'gaussian': {'var': 0.01},
            'sp': {'amount': 0.05},
            'speckle': {'var': 0.04},
            'poisson': {'levels': 256},
            'mixed_poisson_gaussian': {'gaussian_var': 0.01}
        },
        'denoising': {
            'median': {'ksize': 3},
            'gaussian': {'ksize': [3, 3], 'sigma': 1.0},
            'wiener': {'mysize': 3},
            'bilateral': {'d': 5, 'sigma_color': 50.0, 'sigma_space': 50.0},
            'non_local_means': {'h': 5, 'template_window_size': 3, 'search_window_size': 11},
            'anscombe_wiener': {'balance': 0.1},
            'adaptive_median': {'s_max': 5},
            'kuan': {'win_size': 3, 'noise_var_estimate': 0.04}
        }
    }

def test_five_noise_types(sample_image, config):
    noise_types = ['gaussian', 's&p', 'speckle', 'poisson', 'mixed_poisson_gaussian']
    for n_type in noise_types:
        noisy_img = add_noise(sample_image, n_type, config)
        assert noisy_img.shape == sample_image.shape, f"Failed shape for noise {n_type}"
        assert not np.isnan(noisy_img).any(), f"NaN found in noise {n_type}"
        assert 0.0 <= noisy_img.min() and noisy_img.max() <= 1.0

def test_eight_denoising_methods(sample_image, config):
    denoising_methods = ['median', 'gaussian', 'wiener', 'bilateral', 'non_local_means', 'anscombe_wiener', 'adaptive_median', 'kuan']
    noisy_img = add_noise(sample_image, 'gaussian', config)
    for method in denoising_methods:
        denoised_img = apply_denoising(noisy_img, method, config)
        assert denoised_img.shape == sample_image.shape, f"Failed shape for denoising {method}"
        assert not np.isnan(denoised_img).any(), f"NaN found in denoising {method}"
        assert 0.0 <= denoised_img.min() and denoised_img.max() <= 1.0

def test_model_forward_pass():
    batch_size = 4
    model = CoreNeedleBiopsyClassifier(backbone="resnet50", num_classes=2, pretrained=False)
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (batch_size, 2), f"Expected output shape (4, 2), got {output.shape}"

def test_metrics_calculation():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.7, 0.3]])

    metrics = calculate_classification_metrics(y_true, y_pred, y_probs)
    assert 'Accuracy' in metrics
    assert 'Precision' in metrics
    assert 'Recall' in metrics
    assert 'F1-Score' in metrics
    assert 'ROC-AUC' in metrics
    assert 'Confusion_Matrix' in metrics
    assert metrics['Accuracy'] == 5.0 / 6.0

def test_checkpoint_saving_and_loading(tmp_path):
    save_dir = str(tmp_path)
    model = CoreNeedleBiopsyClassifier(backbone="resnet50", num_classes=2, pretrained=False)
    save_checkpoint({'epoch': 5, 'state_dict': model.state_dict(), 'best_acc': 0.95}, save_dir=save_dir, filename="test_ckpt.pt")
    
    ckpt_file = os.path.join(save_dir, "test_ckpt.pt")
    assert os.path.exists(ckpt_file)
    
    new_model = CoreNeedleBiopsyClassifier(backbone="resnet50", num_classes=2, pretrained=False)
    epoch, best_acc = load_checkpoint(ckpt_file, new_model)
    assert epoch == 5
    assert best_acc == 0.95
