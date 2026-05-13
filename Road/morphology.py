import cv2
import numpy as np

def prepare_mask_for_edges(binary_mask, kernel_size=5, min_area=200):
    """
    Applies morphological operations to the binary mask BEFORE edge detection.
    
    Order: Erosion → Opening → Island Removal → Closing
    - Erosion:        Shaves the white (crack) regions, removing thin noise bridges
                      and shrinking small blobs below the kernel threshold.
    - Opening:        Erodes then dilates — completely wipes out any isolated specks
                      that survived the first erosion, then restores the surviving shapes.
    - Island Removal: Labels every white blob via connected components and deletes any
                      whose area (in pixels) is below min_area. More surgical than morphology.
    - Closing:        Bridges small gaps between crack segments split by the steps above.

    Args:
        binary_mask: Black-and-white mask from thresholding (white = crack).
        kernel_size: Size of the square structuring element. Larger = more aggressive.
        min_area:    Minimum blob size (pixels) to keep. Blobs smaller than this are
                     treated as noise islands and erased.
                     Lower  → keeps more small regions.
                     Higher → only large crack blobs survive.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Step 1: EROSION — trim the white crack regions and kill thin noise bridges
    #
    # eroded = cv2.erode(binary_mask, kernel, iterations=1)

    # Step 2: OPENING (Erosion → Dilation) — remove any remaining isolated specks
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Step 3: ISLAND REMOVAL via connected components
    # Label every distinct white blob; delete those whose area < min_area.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    for label in range(1, num_labels):  # label 0 is the background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label] = 255

    # Step 4: CLOSING (Dilation → Erosion) — bridge small gaps between crack segments
    # that may have been split apart by the erosion/opening steps above.
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    return closed