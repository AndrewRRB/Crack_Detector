# import cv2
# import numpy as np

# def clean_crack_mask(edge_map, kernel_size=5):
#     """
#     Thickens the thin Canny edges and merges them into a solid mask.
#     """
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
#     # Step 1: DILATION (The Thickener)
#     # We force those 1-pixel thin Canny edges to expand outward. 
#     # Because a crack usually has two parallel edges (left and right boundary),
#     # dilating them will cause them to crash into each other and fill the middle.
#     thick_edges = cv2.dilate(edge_map, kernel, iterations=1)
    
#     # Step 2: CLOSING (The Gap Bridger)
#     # Dilation -> Erosion to smooth out the newly thickened mask 
#     # and seal up any remaining micro-gaps along the length of the crack.
#     final_clean_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel)
    
#     return final_clean_edges

import cv2
import numpy as np

def clean_crack_mask(edge_map, kernel_size=5):
    """
    Thickens the thin Canny edges, merges them, and fills the interior into a solid mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. THE THICKENER (Increased to 3 iterations!)
    # We force the left and right boundaries of the crack to expand until they 
    # touch each other and form a single, closed loop.
    thick_edges = cv2.dilate(edge_map, kernel, iterations=3)
    
    # 2. THE GAP BRIDGER
    # Smooths out the edges of our newly merged shape.
    closed_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. THE PAINTER (New logic to fill the hollow center)
    # Find the boundary of the closed shape we just made
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a pitch-black canvas
    final_clean_edges = np.zeros_like(closed_edges)
    
    # Loop through the boundaries and paint the inside solid white (thickness=cv2.FILLED)
    for contour in contours:
        cv2.drawContours(final_clean_edges, [contour], -1, 255, thickness=cv2.FILLED)
            
    return final_clean_edges