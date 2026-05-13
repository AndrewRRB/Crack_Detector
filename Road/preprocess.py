import cv2

def load_and_preprocess(image_path, threshold=100):
    """Loads an image, converts to grayscale, reduces noise, and applies binary thresholding.
    
    Args:
        image_path: Path to the input image.
        threshold:  Pixel intensity cutoff (0–255). Pixels darker than this value
                    become WHITE (255) in the mask; brighter pixels become BLACK (0).
                    Lower  → more pixels are treated as cracks (more sensitive).
                    Higher → only the darkest pixels are treated as cracks (stricter).
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bro, I can't find the image at {image_path}. Check your path!")
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Spatial Filtering (Bilateral Blur)
    # Preserves edges better than Gaussian Blur — great for crack boundaries.
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Binary Thresholding → produces a black-and-white mask
    # THRESH_BINARY_INV: dark pixels (cracks) → white (255), bright background → black (0).
    # Tune `threshold` to control sensitivity.
    _, binary_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    return img, binary_mask, blurred