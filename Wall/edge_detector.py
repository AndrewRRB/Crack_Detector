import cv2

def detect_edges(blurred_image, low_threshold=50, high_threshold=150):
    """
    Applies Canny edge detection to find structural cracks.
    - low_threshold: Anything below this gradient intensity is ignored.
    - high_threshold: Anything above this is guaranteed to be an edge.
    - In between: Only kept if it connects to a guaranteed edge.
    """
    # The Canny algorithm highlights the sharp changes in pixel intensity (the cracks)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges