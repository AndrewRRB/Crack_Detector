import cv2
import os
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

# def main():
#     # Grab a sample image from your raw data folder
#     # Make sure you actually drop an image named 'sample_crack.jpg' in there first!
#     raw_dir = os.path.join("..", "data", "raw")
#     output_dir = os.path.join("..", "data", "output")
#     image_path = os.path.join("..", "data", "raw", "00001.jpg") 
    
#     print("[INFO] Firing up the Crack Detection Pipeline...")
    
#     try:
#         # Run Step 1: Preprocessing
#         original, preprocessed = load_and_preprocess(image_path)

#         edges = detect_edges(preprocessed, low_threshold=50, high_threshold=150)
        
#         clean_edges = clean_crack_mask(edges, kernel_size=5)

#         final_visualization = overlay_mask(original, clean_edges)

#         # Display the results
#         cv2.imshow("Original Concrete", original)
#         cv2.imshow("Preprocessed (Grayscale + Blurred)", preprocessed)
#         cv2.imshow("Detected Cracks (Edges)", edges)
#         cv2.imshow("Cleaned Crack Mask", clean_edges)
#         cv2.imshow("Final Visualization (Cracks in Red)", final_visualization)
        
#         print("[INFO] Boom. Images loaded. Press any key on the image window to close.")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"[ERROR] {e}")

import cv2
import os
import numpy as np
from preprocess import load_and_preprocess
from edge_detector import detect_edges  # Updated to match your filename!
from morphology import clean_crack_mask

def overlay_mask(original, mask):
    """Paints the white crack mask bright red on the original image."""
    color_mask = np.zeros_like(original)
    color_mask[mask == 255] = [0, 0, 255] # BGR for Red
    overlaid_img = cv2.addWeighted(original, 1.0, color_mask, 0.6, 0)
    return overlaid_img

def main():
    raw_dir = os.path.join("..", "data", "raw")
    output_dir = os.path.join("..", "processed")
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] Firing up the Full Dataset Scanner...")
    
    # We use os.walk so it automatically searches inside Positive/Negative subfolders!
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            image_path = os.path.join(root, filename)
            
            try:
                # 1. Preprocessing
                original, preprocessed = load_and_preprocess(image_path)

                # 2. Edge Detection
                edges = detect_edges(preprocessed, low_threshold=40, high_threshold=120)
                
                # 3. Morphology Cleanup & Sniper
                # (Ensure clean_crack_mask in morphology.py has the filled contour logic!)
                clean_edges = clean_crack_mask(edges, kernel_size=5)

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