import cv2
import os
import sys
import numpy as np 
from preprocess import load_and_preprocess
from edge_detector import detect_edges
from morphology import clean_crack_mask


def overlay_mask(original, mask):
    """Paints the white crack mask bright red on the original image."""
    color_mask = np.zeros_like(original)
    color_mask[mask == 255] = [0, 0, 255]
    overlaid_img = cv2.addWeighted(original, 1.0, color_mask, 0.6, 0)
    return overlaid_img


def main(gui_image_path=None):
    # Grab the image path passed from the GUI
    if gui_image_path is None and len(sys.argv) > 1:
        gui_image_path = sys.argv[1]

    # Safety check: Did the GUI actually send a valid file?
    if not gui_image_path or not os.path.exists(gui_image_path):
        print("[ERROR] No valid image path provided from GUI.")
        return

    output_dir = os.path.join("..", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(gui_image_path)
    print(f"[INFO] Firing up the Crack Detection Pipeline for: {filename}")
    
    try:
        # 1. Preprocessing
        original, preprocessed = load_and_preprocess(gui_image_path)
        cv2.imshow("Preprocessed Image", preprocessed)

        # 2. Edge Detection
        edges = detect_edges(preprocessed, low_threshold=10, high_threshold=80)
        cv2.imshow("Edges", edges)
        
        # 3. Morphology Cleanup
        clean_edges = clean_crack_mask(edges, kernel_size=3)
        cv2.imshow("Clean Edges", clean_edges)
        
        # 4. LOGIC CHECK: Is there actually a crack?
        crack_pixels = cv2.countNonZero(clean_edges)
        cv2.imshow("Crack Mask", clean_edges)

        if crack_pixels > 0:
            print(f"[DETECTED] Crack found in {filename} ({crack_pixels}px)")
            final_visual = overlay_mask(original, clean_edges)
            cv2.putText(final_visual, "DEFECT DETECTED", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            print(f"[CLEAR] No cracks in {filename}")
            final_visual = original.copy()
            cv2.putText(final_visual, "SURFACE CLEAR", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save the final image to the output folder
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(output_path, final_visual)
        print(f"[INFO] Processing complete. Result saved to: {output_path}")
        
        # Display it on screen for the user
        cv2.imshow("Crack Detector System", final_visual)
        print("[INFO] Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] Failed on {filename}: {e}")

if __name__ == "__main__":
    # Fix the working directory issue if you run this from the root folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()