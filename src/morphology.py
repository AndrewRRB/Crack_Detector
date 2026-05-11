# import cv2
# import numpy as np

# def clean_crack_mask(edge_map, kernel_size=5):
#     """
#     Uses morphological transformations to bridge gaps and remove noise.
#     """
#     # Create the structural element (a 5x5 grid of 1s)
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
#     # Step 1: CLOSING (Dilation followed by Erosion)
#     # This expands the white pixels to bridge the tiny gaps between crack segments, 
#     # then shrinks them back down to their normal thickness.
#     closed_edges = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
    
#     # Step 2: OPENING (Erosion followed by Dilation)
#     # This completely erodes away tiny, isolated noise specks (like pebbles).
#     # Since they disappear completely, the subsequent dilation can't bring them back.
#     final_clean_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)
    
#     return final_clean_edges
import cv2
import numpy as np

def clean_crack_mask(edge_map, kernel_size=5, min_area=500):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Step 1: Thicken and Connect
    # Dilate the thin Canny lines, then Close them to bridge gaps
    thick_edges = cv2.dilate(edge_map, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel)
    
    # Step 2: The Contour Sniper
    # Find every independent white blob on the screen
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a pitch-black canvas
    final_clean_edges = np.zeros_like(closed_edges)
    
    # Loop through the blobs. If it's huge, draw it. If it's small, ignore it.
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(final_clean_edges, [contour], -1, 255, thickness=cv2.FILLED)
            
    return final_clean_edges