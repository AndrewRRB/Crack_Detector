import cv2
import os
import sys
import numpy as np
from preprocess import load_and_preprocess
from edge_detector import detect_edges
from morphology import prepare_mask_for_edges


def overlay_mask(original, mask):
    """Paints the white crack mask bright red on the original image."""
    color_mask = np.zeros_like(original)
    color_mask[mask == 255] = [0, 0, 255]
    overlaid_img = cv2.addWeighted(original, 1.0, color_mask, 0.6, 0)
    return overlaid_img

def main(image_path=None):
    # Image path is passed in from the GUI launcher
    if image_path is None:
        if len(sys.argv) < 2:
            print("[ERROR] No image path provided. Run via launcher.py.")
            return
        image_path = sys.argv[1]
    
    print("[INFO] Firing up the Crack Detection Pipeline...")
    
    try:
        # Step 1: go to the preprocess.py file to load and preprocess the image
        original, binary_mask, blurred = load_and_preprocess(image_path)

        # Step 2: Morphology on the binary mask (Erosion → Opening)
        cleaned_mask = prepare_mask_for_edges(binary_mask, kernel_size=5, min_area=300)

        # Step 3: Canny edge detection on the cleaned mask
        edges = detect_edges(cleaned_mask, low_threshold=40, high_threshold=80)

        # Step 4: LOGIC CHECK: Is there actually a crack?
        crack_pixels = cv2.countNonZero(edges)
        
        if crack_pixels > 0:
            print(f"[DETECTED] Crack found ({crack_pixels}px)")
            final_visual = overlay_mask(original, edges)
            cv2.putText(final_visual, "DEFECT DETECTED", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            print("[CLEAR] No cracks detected.")
            final_visual = original.copy()
            cv2.putText(final_visual, "SURFACE CLEAR", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        
        # Display all pipeline stages
        cv2.imshow("Original Concrete", original)
        cv2.imshow("Grayscale", blurred)
        cv2.imshow("Binary Mask (after Threshold)", binary_mask)
        cv2.imshow("Cleaned Mask (after Erosion + Opening)", cleaned_mask)
        cv2.imshow("Final Edges (Canny)", edges)
        cv2.imshow("Crack Detector System", final_visual)
        
        print("[INFO] Boom. Images loaded. Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    # Fix the working directory issue if you run this from the root folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
