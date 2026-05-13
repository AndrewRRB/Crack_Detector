import cv2
import numpy as np

#    Applies morphological operations to the binary mask BEFORE edge detection.

def prepare_mask_for_edges(binary_mask, kernel_size, min_area):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 1. OPENING (Erosion → Dilation) — remove any remaining isolated specks

    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 2. ISLAND REMOVAL via connected components
    # Label every distinct white blob; delete those whose area < min_area.

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label] = 255

    # 3. CLOSING (Dilation → Erosion) — bridge small gaps between crack segments
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    return closed