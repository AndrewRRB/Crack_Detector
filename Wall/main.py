import cv2
import os
import sys
import numpy as np # Make sure this is at the top of main.py!
from preprocess import load_and_preprocess
from edge_detector import detect_edges
from morphology import clean_crack_mask

# from marr_hildreth import detect_edges_mh

def overlay_mask(original, mask):
    """Paints the white crack mask bright red on the original image."""
    # Create a blank image of the same size
    color_mask = np.zeros_like(original)
    
    # Where the mask is white (255), make the color mask RED (BGR format: 0, 0, 255)
    color_mask[mask == 255] = [0, 0, 255]
    
    # Blend the original image and the red mask together (0.6 opacity for the red)
    overlaid_img = cv2.addWeighted(original, 1.0, color_mask, 0.6, 0)
    return overlaid_img


def main(gui_image_path=None):
    if gui_image_path is None and len(sys.argv) > 1:
        gui_image_path = sys.argv[1]

    raw_dir = os.path.join("..", "data", "raw")
    output_dir = os.path.join("..", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    if gui_image_path:
        print(f"[INFO] Firing up the Crack Detection Pipeline for: {gui_image_path}")
        walk_iter = [(os.path.dirname(gui_image_path), [], [os.path.basename(gui_image_path)])]
    else:
        print("[INFO] Firing up the Full Dataset Scanner...")
        walk_iter = os.walk(raw_dir)
    
    # We use os.walk so it automatically searches inside Positive/Negative subfolders!
    for root, dirs, files in walk_iter:
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            image_path = os.path.join(root, filename)
            
            try:
                # 1. Preprocessing
                original, preprocessed = load_and_preprocess(image_path)

                # 2. Edge Detection
                edges = detect_edges(preprocessed, low_threshold=10, high_threshold=80)
                
                # 3. Morphology Cleanup & Sniper
                # (Ensure clean_crack_mask in morphology.py has the filled contour logic!)
                clean_edges = clean_crack_mask(edges, kernel_size=3)

                # 4. LOGIC CHECK: Is there actually a crack?
                # countNonZero counts how many white pixels survived the morphology cleanup
                crack_pixels = cv2.countNonZero(clean_edges)
                
                if crack_pixels > 0:
                    print(f"[DETECTED] Crack found in {filename} ({crack_pixels}px)")
                    # Paint the crack red
                    final_visual = overlay_mask(original, clean_edges)
                    # Add a red warning label
                    cv2.putText(final_visual, "DEFECT DETECTED", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    print(f"[CLEAR] No cracks in {filename}")
                    # Leave the original image alone, but add a green clear label
                    final_visual = original.copy()
                    cv2.putText(final_visual, "SURFACE CLEAR", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Save the final image to the output folder
                output_path = os.path.join(output_dir, f"result_{filename}")
                cv2.imwrite(output_path, final_visual)
                
                # Display it on screen for a split second (Press 'q' to quit early)
                cv2.imshow("Crack Detector System", final_visual)
                if gui_image_path:
                    print("[INFO] Press any key on the image window to close.")
                    cv2.waitKey(0)
                else:
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        print("[INFO] Run aborted by user.")
                        cv2.destroyAllWindows()
                        return
                
            except Exception as e:
                print(f"[ERROR] Failed on {filename}: {e}")

    cv2.destroyAllWindows()
    print("[INFO] Batch processing complete. All results saved to the 'output' folder!")



if __name__ == "__main__":
    # Fix the working directory issue if you run this from the root folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()