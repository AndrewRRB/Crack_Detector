import cv2
import numpy as np

def clean_crack_mask(edge_map, kernel_size=5, min_area=500):
    """
    Thickens Canny edges and deletes small noise (pebbles) using contour area.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. Thicken the edges so the crack connects into a solid shape
    thick_edges = cv2.dilate(edge_map, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel)
    
    # 2. CONTOUR FILTERING (The Sniper)
    # Find all distinct white shapes in the image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a pitch-black canvas to draw our final results on
    final_clean_edges = np.zeros_like(closed_edges)
    
    # Loop through every shape we found
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # If the shape is massive (like our crack), draw it in solid white
        # If it's small (like a pebble shadow), ignore it completely
        if area > min_area:
            cv2.drawContours(final_clean_edges, [contour], -1, 255, thickness=cv2.FILLED)
            
    return final_clean_edges