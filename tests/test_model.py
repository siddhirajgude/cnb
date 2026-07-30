import torch
import unittest
import numpy as np
from src.utils import load_config
from src.model_gpu import add_noise_gpu, apply_denoising_gpu

class TestMammographyPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a mock configuration
        self.config = {
            'noise': {
                'gaussian': {'var': 0.01},
                'sp': {'amount': 0.05},
                'speckle': {'var': 0.04},
                'mixed_poisson_gaussian': {'gaussian_var': 0.01}
            },
            'denoising': {
                'median': {'kernel_size': 5},
                'gaussian': {'kernel_size': [5, 5], 'sigma': 0},
                'wiener': {'balance': 0.1},
                'bilateral': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
                'non_local_means': {'h': 10, 'template_window_size': 7, 'search_window_size': 21},
                'adaptive_median': {'s_max': 7},
                'gabor': {
                    'ksize': 15,
                    'sigma': 3.0,
                    'lambd': 5.0,
                    'gamma': 0.5,
                    'psi': 0.0,
                    'orientations': [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
                }
            }
        }
        # Create a mock grayscale image tensor (B=1, C=1, H=256, W=256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_img = torch.rand(1, 1, 256, 256, dtype=torch.float32, device=self.device)

    def test_noise_addition(self):
        noise_types = ['gaussian', 's&p', 'speckle', 'poisson', 'mixed_poisson_gaussian']
        for noise in noise_types:
            noisy_img = add_noise_gpu(self.test_img, noise, self.config, self.device)
            self.assertEqual(noisy_img.shape, self.test_img.shape, f"Shape mismatch for noise: {noise}")
            self.assertTrue(torch.all(noisy_img >= 0.0), f"Negative pixel values for noise: {noise}")

    def test_denoising_methods(self):
        denoising_methods = ['median', 'gaussian', 'wiener', 'bilateral', 'non_local_means', 'anscombe_wiener', 'adaptive_median', 'kuan', 'gabor']
        # Apply gaussian noise to create a noisy image
        noisy_img = add_noise_gpu(self.test_img, 'gaussian', self.config, self.device)
        
        for method in denoising_methods:
            denoised_img = apply_denoising_gpu(noisy_img, method, self.config)
            self.assertEqual(denoised_img.shape, self.test_img.shape, f"Shape mismatch for method: {method}")
            self.assertTrue(torch.all(denoised_img >= 0.0), f"Negative pixel values for method: {method}")
            self.assertTrue(torch.all(denoised_img <= 1.0), f"Pixel values exceeding 1.0 for method: {method}")

if __name__ == '__main__':
    unittest.main()
