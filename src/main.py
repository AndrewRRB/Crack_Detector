# import cv2
# import os
# from preprocess import load_and_preprocess
# from edge_detector import detect_edges
# from morphology import clean_crack_mask

# # from marr_hildreth import detect_edges_mh

# def main():
#     # Grab a sample image from your raw data folder
#     # Make sure you actually drop an image named 'sample_crack.jpg' in there first!
#     image_path = os.path.join("..", "data", "raw", "road.jpg") 
    
#     print("[INFO] Firing up the Crack Detection Pipeline...")
    
#     try:
#         # Run Step 1: Preprocessing
#         original, preprocessed = load_and_preprocess(image_path)

#         edges = detect_edges(preprocessed, low_threshold=50, high_threshold=150)
        
#         clean_edges = clean_crack_mask(edges, kernel_size=5)

#         # Display the results
#         cv2.imshow("Original Concrete", original)
#         cv2.imshow("Preprocessed (Grayscale + Blurred)", preprocessed)
#         cv2.imshow("Detected Cracks (Edges)", edges)
#         cv2.imshow("Cleaned Crack Mask", clean_edges)
        
#         print("[INFO] Boom. Images loaded. Press any key on the image window to close.")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"[ERROR] {e}")

# if __name__ == "__main__":
#     # Fix the working directory issue if you run this from the root folder
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(script_dir)
    
#     main()
import cv2
import os
from preprocess import load_and_preprocess
from edge_detector import detect_edges
from morphology import clean_crack_mask

def main():
    # Pointing to the new asphalt image
    
    image_path = os.path.join("..", "data", "raw", "road.jpg") 
    
    print("[INFO] Firing up the Asphalt Crack Pipeline...")
    
    try:
        # 1. Preprocessing (Grayscale + Median Blur)
        original, preprocessed = load_and_preprocess(image_path)
        
        # 2. Canny Edge Detection
        # You might need to tweak these! If the crack isn't showing, lower them.
        raw_edges = detect_edges(preprocessed, low_threshold=40, high_threshold=120)
        
        # 3. Morphology & Area Filtering
        # If pieces of your crack are getting deleted, lower min_area (e.g., to 200)
        # If rocks are still surviving, raise min_area (e.g., to 1000)
        final_mask = clean_crack_mask(raw_edges, kernel_size=5, min_area=500)
        
        # Display the results
        cv2.imshow("1. Original Asphalt", original)
        cv2.imshow("2. Median Blurred", preprocessed)
        cv2.imshow("3. Canny Edges", raw_edges)
        cv2.imshow("4. Final Clean Mask", final_mask)
        
        print("[INFO] Done. Press any key on an image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()