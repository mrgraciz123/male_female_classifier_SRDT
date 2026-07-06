import os
import sys
import pickle
import numpy as np
import cv2

# Configuration
MODEL_FILENAME = "gender_model.pkl"
IMAGE_SIZE = (64, 64)

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Reads an image from a path, resizes it, converts it to grayscale,
    flattens it, and normalizes it, matching the training preprocessing pipeline.
    
    Parameters:
        image_path (str): Path to the image file.
        target_size (tuple): Dimensions to resize to.
        
    Returns:
        numpy.ndarray: 2D array of shape (1, flattened_size) suitable for prediction.
    """
    # 1. Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load or find image at: {image_path}")
        
    # 2. Resize the image to 64x64
    img_resized = cv2.resize(img, target_size)
    
    # 3. Convert image to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 4. Flatten the image to a 1D vector
    img_flattened = img_gray.flatten()
    
    # 5. Normalize pixel values to range [0, 1]
    img_normalized = img_flattened.astype(np.float32) / 255.0
    
    # Reshape vector to 2D array of shape (1, num_features) because scikit-learn models
    # expect X to be a 2D matrix (samples, features)
    return img_normalized.reshape(1, -1)

def load_saved_model(model_path=MODEL_FILENAME):
    """
    Loads the trained pickle model from disk.
    
    Parameters:
        model_path (str): Path to the pickle file.
        
    Returns:
        LinearRegression: The loaded scikit-learn LinearRegression model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Error: Model file '{model_path}' not found.\n"
            "Please train the model first by running: python train_model.py"
        )
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    # Verify that an image path is provided
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py dataset/male/063429.jpg.jpg")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    try:
        # 1. Load the model
        model = load_saved_model(MODEL_FILENAME)
        
        # 2. Preprocess the input image
        processed_img = preprocess_image(image_path)
        
        # 3. Predict the raw continuous output
        prediction_val = model.predict(processed_img)[0]
        
        # 4. Apply the classification threshold (0.5)
        # prediction >= 0.5 -> Male
        # prediction < 0.5 -> Female
        if prediction_val >= 0.5:
            gender_class = "Male"
            confidence = prediction_val
        else:
            gender_class = "Female"
            confidence = 1 - prediction_val
            
        print("\n" + "="*40)
        print("PREDICTION RESULT")
        print("="*40)
        print(f"Image Path:       {image_path}")
        print(f"Raw Output Value: {prediction_val:.4f}")
        print(f"Predicted Gender: {gender_class}")
        # Note: Linear regression raw predictions can occasionally exceed [0, 1] bounds,
        # so we clip or simply show raw value.
        print("="*40)
        
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
