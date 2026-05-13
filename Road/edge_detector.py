import cv2

# The Canny algorithm highlights the sharp changes in pixel intensity (the cracks)

def detect_edges(blurred_image, low_threshold, high_threshold):
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges