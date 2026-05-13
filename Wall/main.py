import cv2
import os
import sys
from preprocess import load_and_preprocess
from edge_detector import detect_edges
from morphology import clean_crack_mask

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
        # Run Step 1: Preprocessing
        original, preprocessed = load_and_preprocess(image_path)

        edges = detect_edges(preprocessed, low_threshold=50, high_threshold=90)
        
        clean_edges = clean_crack_mask(edges, kernel_size=5)

        # Display the results
        cv2.imshow("Original Concrete", original)
        cv2.imshow("Preprocessed (Grayscale + Blurred)", preprocessed)
        cv2.imshow("Detected Cracks (Edges)", edges)
        cv2.imshow("Cleaned Crack Mask", clean_edges)
        
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