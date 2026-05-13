import cv2
import os
import sys
from preprocess import load_and_preprocess
from edge_detector import detect_edges
from morphology import prepare_mask_for_edges

# from marr_hildreth import detect_edges_mh

def main(image_path=None):
    # Image path is passed in from the GUI launcher via sys.argv.
    if image_path is None:
        if len(sys.argv) < 2:
            print("[ERROR] No image path provided. Run via launcher.py.")
            return
        image_path = sys.argv[1]
    
    print("[INFO] Firing up the Crack Detection Pipeline...")
    
    try:
        # Step 1: Load → Grayscale → Bilateral Blur → Binary Threshold
        original, binary_mask, blurred = load_and_preprocess(image_path)

        # Step 2: Morphology on the binary mask (Erosion → Opening)
        # Clean up the mask BEFORE feeding it to the edge detector.
        cleaned_mask = prepare_mask_for_edges(binary_mask, kernel_size=5)

        # Step 3: Canny edge detection on the cleaned mask
        edges = detect_edges(cleaned_mask, low_threshold=50, high_threshold=90)

        # Fill holes
        
        # Display all pipeline stages
        cv2.imshow("Original Concrete", original)
        cv2.imshow("Grayscale", blurred)
        cv2.imshow("Binary Mask (after Threshold)", binary_mask)
        cv2.imshow("Cleaned Mask (after Erosion + Opening)", cleaned_mask)
        cv2.imshow("Final Edges (Canny)", edges)
        
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
