import os
import pickle
import numpy as np
import cv2
import streamlit as st

# Configuration
MODEL_FILENAME = "gender_model.pkl"
IMAGE_SIZE = (64, 64)

# Page configuration
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="👫",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
    }
    .male-card {
        background-color: #e0f2fe;
        border: 2px solid #0ea5e9;
        color: #0369a1;
    }
    .female-card {
        background-color: #fce7f3;
        border: 2px solid #f43f5e;
        color: #be123c;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_model(model_path=MODEL_FILENAME):
    """
    Loads the trained pickle model from disk and caches it to prevent reloading.
    """
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_uploaded_image(uploaded_file, target_size=IMAGE_SIZE):
    """
    Reads an uploaded file buffer, decodes it with OpenCV, resizes it,
    converts it to grayscale, flattens it, and normalizes it.
    """
    # Convert file buffer to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode image using OpenCV
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image.")
        
    # Identical preprocessing steps as training/inference scripts
    img_resized = cv2.resize(img, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flattened = img_gray.flatten()
    img_normalized = img_flattened.astype(np.float32) / 255.0
    
    return img, img_gray, img_normalized.reshape(1, -1)

def main():
    st.title("👫 Male vs Female Image Classifier")
    st.markdown(r"""
        Upload a portrait image, and the Linear Regression model will classify it 
        using the threshold: **Prediction Value $\ge 0.5 \rightarrow$ Male**, else **Female**.
    """)
    
    # Check if model exists
    model = load_saved_model(MODEL_FILENAME)
    if model is None:
        st.error(
            f"Model file '{MODEL_FILENAME}' not found. "
            "Please train the model first by running `python train_model.py` or "
            "executing all cells in `train_model.ipynb`."
        )
        return

    # Image Uploader
    uploaded_file = st.file_uploader(
        "Choose an image file...", 
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded_file is not None:
        try:
            # Preprocess the image
            orig_img, gray_img, processed_vector = preprocess_uploaded_image(uploaded_file)
            
            # Create two columns for display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                # Streamlit takes RGB, OpenCV decodes BGR, so swap channels for correct colors
                st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            with col2:
                st.subheader("Preprocessed (64x64 Grayscale)")
                # Show the resized, grayscaled image that the model actually evaluates
                st.image(gray_img, width=150, use_container_width=False, caption="Model input size")

            # Classification
            if st.button("Classify Image"):
                with st.spinner("Analyzing features..."):
                    # Predict using Linear Regression
                    prediction_val = model.predict(processed_vector)[0]
                    
                    # Classification decision
                    if prediction_val >= 0.5:
                        gender_class = "Male"
                        card_class = "male-card"
                        icon = "👨"
                    else:
                        gender_class = "Female"
                        card_class = "female-card"
                        icon = "👩"
                    
                    # Output Result Card
                    st.markdown(
                        f'<div class="result-card {card_class}">'
                        f'<h2>Predicted Class: {icon} {gender_class}</h2>'
                        f'<p style="font-size: 18px;"><b>Model Output Value:</b> {prediction_val:.4f}</p>'
                        f'<p style="font-size: 14px; margin-top: 5px;">'
                        f'Threshold constraint: Value &ge; 0.5 is Male, &lt; 0.5 is Female'
                        f'</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
