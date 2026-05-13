import cv2

#Loads an image, converts to grayscale, reduces noise, and applies binary thresholding.

def load_and_preprocess(image_path, threshold=100):
    
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bro, I can't find the image at {image_path}. Check your path!")
    
    # 2. Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Spatial Filtering (Bilateral Blur
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Binary Thresholding 
    _, binary_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    return img, binary_mask, blurred