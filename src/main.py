import cv2
import os
from preprocess import load_and_preprocess

def main():
    # Grab a sample image from your raw data folder
    # Make sure you actually drop an image named 'sample_crack.jpg' in there first!
    image_path = os.path.join("..", "data", "raw", "00001.jpg") 
    
    print("[INFO] Firing up the Crack Detection Pipeline...")
    
    try:
        # Run Step 1: Preprocessing
        original, preprocessed = load_and_preprocess(image_path)
        
        # Display the results
        cv2.imshow("Original Concrete", original)
        cv2.imshow("Preprocessed (Grayscale + Blurred)", preprocessed)
        
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