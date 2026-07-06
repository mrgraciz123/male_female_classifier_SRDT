import os
import pickle
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# Configuration
DATASET_DIR = "dataset"
MODEL_FILENAME = "gender_model.pkl"
IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Reads an image using OpenCV, resizes it, converts it to grayscale,
    flattens it into a 1D vector, and normalizes pixel values.
    
    Parameters:
        image_path (str): Path to the image file.
        target_size (tuple): Target dimensions (width, height) for resizing.
        
    Returns:
        numpy.ndarray: Preprocessed 1D image feature vector.
    """
    # 1. Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at: {image_path}")
    
    # 2. Resize the image to target dimensions (64x64)
    img_resized = cv2.resize(img, target_size)
    
    # 3. Convert image to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 4. Flatten the image into a 1D feature vector
    img_flattened = img_gray.flatten()
    
    # 5. Normalize pixel values to range [0, 1] for stable linear regression training
    img_normalized = img_flattened.astype(np.float32) / 255.0
    
    return img_normalized

def load_dataset(dataset_dir, target_size=IMAGE_SIZE):
    """
    Traverses the dataset directory, loads and preprocesses all images,
    and encodes labels (Male = 1, Female = 0) in parallel using thread pool.
    
    Parameters:
        dataset_dir (str): Root path to the dataset folder.
        target_size (tuple): Target image dimensions.
        
    Returns:
        tuple: (X, y) where X is a numpy array of features and y is a numpy array of labels.
    """
    # We support both lowercase and uppercase directory names
    category_map = {
        'male': 1,
        'female': 0,
        'Male': 1,
        'Female': 0
    }
    
    print(f"Scanning dataset directory: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        
    # Gather all file paths and categories to process
    tasks = []
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        
        if os.path.isdir(category_path) and category in category_map:
            label = category_map[category]
            image_names = os.listdir(category_path)
            for img_name in image_names:
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    img_path = os.path.join(category_path, img_name)
                    tasks.append((img_path, label))
                    
    total_images = len(tasks)
    print(f"Found {total_images} images. Loading and preprocessing in parallel...")
    
    features = []
    labels = []
    
    def process_single_task(task):
        img_path, label = task
        try:
            feat = preprocess_image(img_path, target_size)
            return feat, label
        except Exception:
            # Silently skip corrupted files to keep the console clean during bulk loading
            return None
            
    # Load images in parallel (using 8 threads to balance disk I/O and CPU bound grayscaling/resizing)
    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_single_task, tasks)
        
        for idx, result in enumerate(results):
            if result is not None:
                features.append(result[0])
                labels.append(result[1])
                success_count += 1
            if (idx + 1) % 5000 == 0:
                print(f"Processed {idx + 1}/{total_images} images...")
                
    print(f"Successfully loaded {success_count} images out of {total_images} total images.")
    return np.array(features), np.array(labels)

def train_and_save_model():
    """
    Orchestrates the model training workflow: loads dataset, splits it,
    trains the Linear Regression model, evaluates it, and saves the weights.
    """
    try:
        # Load features and labels
        X, y = load_dataset(DATASET_DIR)
        
        if len(X) == 0:
            print("Error: No images were loaded. Please check your dataset folder structure.")
            return
            
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {X.shape[0]}")
        print(f"Features dimension: {X.shape[1]}")
        
        # Convert to Pandas DataFrame for a brief inspect if desired, or directly split
        # We can create a quick distribution check using Pandas
        df_labels = pd.Series(y)
        print("Label Distribution:")
        print(df_labels.value_counts().rename({1: "Male (1)", 0: "Female (0)"}))
        
        # Split dataset (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        print("\nTraining Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred_continuous = model.predict(X_train)
        y_test_pred_continuous = model.predict(X_test)
        
        # Apply the threshold >= 0.5 to convert regression predictions to classes
        y_train_pred_class = (y_train_pred_continuous >= 0.5).astype(int)
        y_test_pred_class = (y_test_pred_continuous >= 0.5).astype(int)
        
        # Calculate evaluation metrics
        train_r2 = r2_score(y_train, y_train_pred_continuous)
        test_r2 = r2_score(y_test, y_test_pred_continuous)
        
        train_accuracy = accuracy_score(y_train, y_train_pred_class)
        test_accuracy = accuracy_score(y_test, y_test_pred_class)
        
        print("\n" + "="*40)
        print("MODEL EVALUATION RESULTS")
        print("="*40)
        print(f"Train R² Score:                  {train_r2:.4f}")
        print(f"Test R² Score:                   {test_r2:.4f}")
        print(f"Train Classification Accuracy:   {train_accuracy * 100:.2f}%")
        print(f"Test Classification Accuracy:    {test_accuracy * 100:.2f}%")
        print("="*40)
        
        # Save the model
        print(f"\nSaving the trained model to '{MODEL_FILENAME}'...")
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully! You can now use predict.py for inference.")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == "__main__":
    train_and_save_model()
