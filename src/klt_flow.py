import os
import cv2
import numpy as np

def simulate_motion(img: np.ndarray, tx: float = 5.0, ty: float = 3.0, angle: float = 1.0) -> np.ndarray:
    """
    Simulates translation and rotation on an image using border reflection.
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def compute_klt_optical_flow(image_path: str, results_dir: str) -> None:
    """
    Computes KLT Optical Flow on a simulated frame pair and exports:
      1. Motion vectors overlaid on the source image.
      2. Dense magnitude heatmap created via inverse-distance weighted interpolation.
    """
    # 1. Load image and validate path
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
        
    # Resize image to a reasonable size to prevent massive memory usage on WSIs
    max_dim = 1024
    h_orig, w_orig = img_color.shape[:2]
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_color = cv2.resize(img_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # 2. Simulate synthetic second frame (I2)
    img_gray2 = simulate_motion(img_gray, tx=4.0, ty=2.0, angle=0.5)
    
    # 3. Detect keypoints (goodFeaturesToTrack)
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    p0 = cv2.goodFeaturesToTrack(img_gray, mask=None, **feature_params)
    
    if p0 is None or len(p0) == 0:
        print("Warning: No features found to track.")
        return

    # 4. Calculate Lucas-Kanade Optical Flow (Pyramid LK)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_gray, img_gray2, p0, None, **lk_params)
    
    # Filter for valid points (status == 1)
    status_mask = (st == 1).ravel()
    good_old = p0[status_mask].reshape(-1, 2)
    good_new = p1[status_mask].reshape(-1, 2)

    if len(good_old) == 0:
        print("Warning: All feature points were lost during tracking.")
        return

    # 5. Visualization 1: Draw Motion Vectors
    vectors_img = img_color.copy()
    for (x_old, y_old), (x_new, y_new) in zip(good_old, good_new):
        pt_old = (int(round(x_old)), int(round(y_old)))
        pt_new = (int(round(x_new)), int(round(y_new)))
        
        # Red start point, green motion arrow
        cv2.circle(vectors_img, pt_old, 3, (0, 0, 255), -1)
        cv2.arrowedLine(vectors_img, pt_old, pt_new, (0, 255, 0), 2, tipLength=0.3)

    # 6. Visualization 2: Smooth Interpolated Magnitude Map (calculated on a downsampled grid for speed & safety)
    displacements = good_new - good_old
    magnitudes = np.linalg.norm(displacements, axis=1)

    grid_h, grid_w = 128, 128
    scale_y = grid_h / h
    scale_x = grid_w / w
    good_old_scaled = good_old * [scale_x, scale_y]

    # Create grid of coordinates
    grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
    grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()]).T  # (16384, 2)

    # Fast IDW (Inverse Distance Weighting) interpolation using distance matrix
    dists = np.linalg.norm(grid_pts[:, None, :] - good_old_scaled[None, :, :], axis=2)  # (16384, N_points)
    weights = 1.0 / (dists**2 + 1e-5)  # Avoid division by zero
    weights /= weights.sum(axis=1, keepdims=True)

    # Compute dense magnitude field on the grid and resize to full resolution
    dense_mag_grid = (weights @ magnitudes).reshape(grid_h, grid_w)
    dense_mag = cv2.resize(dense_mag_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 255] and apply Jet Colormap
    mag_norm = cv2.normalize(dense_mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    magnitude_map_colored = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

    # 7. Save Results
    os.makedirs(results_dir, exist_ok=True)
    vectors_path = os.path.join(results_dir, 'klt_motion_vectors.png')
    magnitude_path = os.path.join(results_dir, 'klt_magnitude_map.png')
    
    cv2.imwrite(vectors_path, vectors_img)
    cv2.imwrite(magnitude_path, magnitude_map_colored)
    
    print(f"KLT visualizations successfully saved to '{results_dir}':")
    print(f"  - Motion vectors: {vectors_path}")
    print(f"  - Magnitude map: {magnitude_path}")