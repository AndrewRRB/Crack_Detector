import cv2
import numpy as np

def detect_edges_mh(blurred_image, kernel_size=5):
    """
    Highly optimized Marr-Hildreth (Laplacian of Gaussian) edge detector.
    Uses vectorized morphological operations to find zero-crossings instantly.
    """
    # 1. Calculate the Laplacian (2nd Derivative)
    # We use CV_64F to capture negative values!
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=kernel_size)
    
    # 2. Fast Zero-Crossing Detection using structural shifts
    min_log = cv2.morphologyEx(laplacian, cv2.MORPH_ERODE, np.ones((3,3)))
    max_log = cv2.morphologyEx(laplacian, cv2.MORPH_DILATE, np.ones((3,3)))
    
    # A zero-crossing happens where the min neighbor is negative, and the pixel is positive 
    # OR the max neighbor is positive, and the pixel is negative.
    zero_crossings = np.logical_or(
        np.logical_and(min_log < 0, laplacian > 0),
        np.logical_and(max_log > 0, laplacian < 0)
    )
    
    # Convert boolean array back to a viewable 8-bit image
    edges = np.uint8(zero_crossings) * 255
    return edges