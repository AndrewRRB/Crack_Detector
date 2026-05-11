import cv2

def load_and_preprocess(image_path):
    """Loads an image, converts to grayscale, and reduces noise."""
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bro, I can't find the image at {image_path}. Check your path!")
    
    # 2. Convert to grayscale (color is useless for finding structural cracks)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Spatial Filtering (Gaussian Blur)
    # The (5, 5) is the kernel size. If the concrete is super noisy, bump it to (7, 7).
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.bilateralFilter(gray, 9, 75, 75)  # Better at preserving edges than Gaussian Blur

    return img, blurred