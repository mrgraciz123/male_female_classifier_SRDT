# Male vs Female Image Classifier

An educational college assignment project implementing a binary image classifier to distinguish between **Male** and **Female** portraits. 

This project follows the architecture and coding style of a basic Cats vs Dogs classifier but utilizes **OpenCV** for preprocessing, **NumPy** & **Pandas** for data handling, and scikit-learn's **Linear Regression** model under a threshold-based classification scheme.

---

## Project Description

In this project, image classification is approached through a linear regression lens rather than deep learning or logistic regression. 
Each input image is read in color, resized to $64 \times 64$ pixels, converted to grayscale, and flattened into a $4096$-dimensional feature vector. The feature values are scaled to $[0, 1]$ via normalization.
A linear regression model is trained to output a continuous value between `0` (Female) and `1` (Male). During inference, a threshold of `0.5` is applied:
- $\text{Prediction} \ge 0.5 \rightarrow \text{Male}$
- $\text{Prediction} < 0.5 \rightarrow \text{Female}$

This project provides a beginner-friendly introduction to image classification pipeline elements, data loading, linear models, and feature preprocessing without deep learning complexity.

---

## Project Structure

```text
Male_Female_Classifier/
│
├── dataset/
│     ├── male/            # Folder containing Male images
│     └── female/          # Folder containing Female images
│
├── train_model.ipynb       # Jupyter Notebook to load data, preprocess, and train
├── train_model.py         # Python script to load data, preprocess, and train (Alternative)
├── predict.py             # Script to make predictions on test/external images
├── app.py                 # Streamlit web application for interactive classification
├── gender_model.pkl       # Saved trained Linear Regression model weights (Serialized)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Dataset Format

Organize your images in the `dataset` folder with two subdirectories:
```text
dataset/
├── male/
│     ├── image1.jpg
│     ├── image2.jpg
│     └── ...
└── female/
      ├── image1.jpg
      ├── image2.jpg
      └── ...
```

---

## Installation

1. Ensure you have Python installed (Python 3.8+ is recommended).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Training the Model
You can train the model using either the Jupyter Notebook or the Python script:

#### Option A: Using the Jupyter Notebook
Open and run all cells in `train_model.ipynb` using Jupyter Lab, VS Code, or any other notebook editor.

#### Option B: Using the Python script
Run the script from your terminal:
```bash
python train_model.py
```

**Expected Training Output Example:**
```text
Scanning dataset directory: dataset
Loading images from category 'female' (Label: 0)...
Successfully loaded 500 images from 'female'.
Loading images from category 'male' (Label: 1)...
Successfully loaded 500 images from 'male'.

Dataset loaded successfully!
Total samples: 1000
Features dimension: 4096
Label Distribution:
Male (1)      500
Female (0)    500
dtype: int64

Training Linear Regression model...

========================================
MODEL EVALUATION RESULTS
========================================
Train R² Score:                  0.4851
Test R² Score:                   0.1245
Train Classification Accuracy:   94.50%
Test Classification Accuracy:    71.50%
========================================

Saving the trained model to 'gender_model.pkl'...
Model saved successfully! You can now use predict.py for inference.
```

### 2. Predict on a Custom Image via Command Line
To classify a new image, use the inference script `predict.py` and pass the path to the target image as a command-line argument:
```bash
python predict.py <path_to_image>
```

**Example Command:**
```bash
python predict.py dataset/male/063429.jpg.jpg
```

**Expected Output Example:**
```text
========================================
PREDICTION RESULT
========================================
Image Path:       dataset/male/063429.jpg.jpg
Raw Output Value: 1.0166
Predicted Gender: Male
========================================
```

### 3. Running the Streamlit Web Application
To start the interactive web interface where you can upload images directly in your browser:
```bash
streamlit run app.py
```
This will start a local server and open the application page in your web browser. You can upload an image, view the preprocessed representation, and click "Classify Image" to run inference.

---

## Image Preprocessing Details

Every image undergoes the following sequence of transformations before model ingestion:
1. **Reading**: Loaded via OpenCV `cv2.imread()`.
2. **Resizing**: Downscaled/upscaled to a uniform $64 \times 64$ grid (`cv2.resize()`) to establish a fixed length feature input.
3. **Grayscaling**: Converted to single-channel gray (`cv2.COLOR_BGR2GRAY`) to reduce complexity.
4. **Flattening**: Reshaped into a 1D vector of length $4096 = 64 \times 64 \times 1$.
5. **Normalization**: Scaled by dividing by $255.0$ to maps pixel integers to $[0, 1]$.
6. **Encoding**: Categorized to continuous training targets: `Male` = $1$, `Female` = $0$.
