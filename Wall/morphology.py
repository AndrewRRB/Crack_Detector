import cv2
import numpy as np

def clean_crack_mask(edge_map, kernel_size=5):
    """
    Uses morphological transformations to bridge gaps and remove noise.
    """
    # Create the structural element (a 5x5 grid of 1s)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Step 1: CLOSING (Dilation followed by Erosion)
    # This expands the white pixels to bridge the tiny gaps between crack segments, 
    # then shrinks them back down to their normal thickness.
    closed_edges = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
    
    # Step 2: OPENING (Erosion followed by Dilation)
    # This completely erodes away tiny, isolated noise specks (like pebbles).
    # Since they disappear completely, the subsequent dilation can't bring them back.
    final_clean_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)
    
    return final_clean_edges