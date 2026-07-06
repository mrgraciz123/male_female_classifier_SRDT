import os
import pickle
import time
import base64
import io
import numpy as np
import cv2
import streamlit as st

# Configuration
MODEL_FILENAME = "gender_model.pkl"
IMAGE_SIZE = (64, 64)

# Page configuration
st.set_page_config(
    page_title="Male vs Female AI Classifier",
    page_icon="👫",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Premium CSS for dark mode, animations, custom typography
st.markdown(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    .sidebar-section {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    .sidebar-title {
        color: #6366F1;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .tech-tag {
        display: inline-block;
        background-color: #334155;
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 4px 2px;
        font-weight: 500;
        border: 1px solid #475569;
    }

    /* File uploader container */
    [data-testid="stFileUploader"] {
        background-color: #1E293B !important;
        border: 2px dashed #475569 !important;
        border-radius: 24px !important;
        padding: 24px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366F1 !important;
        transform: translateY(-2px);
    }
    
    /* File uploader button and texts */
    [data-testid="stFileUploader"] button {
        background-color: #6366F1 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #4F46E5 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Hide default streamlit file uploader helper text */
    [data-testid="stFileUploader"] section > div {
        color: #94A3B8 !important;
    }

    /* Rounded image preview */
    .preview-container {
        display: flex;
        justify-content: center;
        margin: 25px 0;
        animation: fadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .preview-img {
        border-radius: 24px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
        border: 2px solid #334155;
        transition: transform 0.3s ease;
        max-width: 100%;
        height: auto;
    }
    .preview-img:hover {
        transform: scale(1.02);
    }

    /* Result Card Styles */
    .result-card-male {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        border: 1px solid #3B82F6;
        border-radius: 24px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.15), 0 10px 10px -5px rgba(59, 130, 246, 0.08);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        margin-top: 24px;
    }

    .result-card-female {
        background: linear-gradient(135deg, #581C87 0%, #0F172A 100%);
        border: 1px solid #D946EF;
        border-radius: 24px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(217, 70, 239, 0.15), 0 10px 10px -5px rgba(217, 70, 239, 0.08);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        margin-top: 24px;
    }

    .gender-badge-male {
        background-color: #3B82F6;
        color: white;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 24px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
    }

    .gender-badge-female {
        background-color: #D946EF;
        color: white;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 24px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(217, 70, 239, 0.3);
    }

    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 25px;
        flex-wrap: wrap;
        gap: 16px;
    }

    .metric-box {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 16px 24px;
        min-width: 140px;
        flex: 1;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 24px;
        font-weight: 700;
    }

    /* Custom Progress Bar */
    .progress-section {
        margin-top: 24px;
        text-align: left;
    }

    .progress-label-container {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        font-size: 14px;
        color: #E2E8F0;
        margin-bottom: 6px;
    }

    .progress-bar-bg {
        background-color: #334155;
        border-radius: 10px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }

    .progress-bar-fill-male {
        background-color: #3B82F6;
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }

    .progress-bar-fill-female {
        background-color: #D946EF;
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #64748B;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #1E293B;
        font-size: 13px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_model(model_path=MODEL_FILENAME):
    """Loads the trained pickle model from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

def preprocess_uploaded_image(uploaded_file, target_size=IMAGE_SIZE):
    """Decodes, resizes, grayscales, flattens, and normalizes the image."""
    # Convert file buffer to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image.")
        
    img_resized = cv2.resize(img, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flattened = img_gray.flatten()
    img_normalized = img_flattened.astype(np.float32) / 255.0
    
    return img, img_normalized.reshape(1, -1)

def calculate_confidence(prediction_val):
    """Maps the linear regression output to a visual confidence score."""
    # Logic: diff of 0.5 represents a clear decision
    # 0.5 is exactly the boundary (50% confidence)
    # Scale: diff * 80% to reach 90% at val = 1.0 or val = 0.0
    diff = abs(prediction_val - 0.5)
    confidence = 50.0 + (diff * 80.0)
    # Clamp confidence value between 51.0% and 99.9%
    return min(max(confidence, 51.0), 99.9)

def main():
    # Sidebar layout
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section">'
            '<div class="sidebar-title">👤 About the AI</div>'
            '<p style="font-size: 14px; color: #94A3B8; margin-bottom: 0;">'
            'This application classifies whether the uploaded portrait belongs to the Male or Female class '
            'using a Linear Regression model trained on grayscale facial images.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            '<div class="sidebar-section">'
            '<div class="sidebar-title">⚙️ Technology Stack</div>'
            '<span class="tech-tag">Python</span>'
            '<span class="tech-tag">Streamlit</span>'
            '<span class="tech-tag">OpenCV</span>'
            '<span class="tech-tag">NumPy</span>'
            '<span class="tech-tag">Scikit-Learn</span>'
            '<span class="tech-tag">Linear Regression</span>'
            '</div>',
            unsafe_allow_html=True
        )

    # Title & Subtitle Header
    st.markdown('<h1 style="text-align: center; margin-bottom: 5px;">👤 Male vs Female AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #94A3B8; font-size: 16px; margin-bottom: 30px;">'
        'Upload a portrait image and let the AI classify it using the trained Linear Regression model.'
        '</p>',
        unsafe_allow_html=True
    )

    # Model Validation
    model = load_saved_model(MODEL_FILENAME)
    if model is None:
        st.error(
            f"Model file '{MODEL_FILENAME}' not found. "
            "Please train the model first by running `python train_model.py` or "
            "executing all cells in `train_model.ipynb`."
        )
        return

    # Image Upload Container
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Save original file bytes and decode
        orig_bytes = uploaded_file.getvalue()
        
        # UI Pre-processing animation
        status_box = st.empty()
        messages = ["Analyzing image...", "Running AI model...", "Processing..."]
        
        # Simulated premium AI delay
        for msg in messages:
            status_box.markdown(f'<p style="text-align: center; color: #6366F1; font-weight: 600;">⏳ {msg}</p>', unsafe_allow_html=True)
            time.sleep(0.35)
        status_box.empty()

        try:
            # Process image
            file_stream = io.BytesIO(orig_bytes)
            orig_img, processed_vector = preprocess_uploaded_image(file_stream)
            
            # Show original image preview styled with custom CSS classes
            encoded_img = base64.b64encode(orig_bytes).decode('utf-8')
            mime_type = "image/png" if uploaded_file.name.lower().endswith("png") else "image/jpeg"
            
            st.markdown(
                f'<div class="preview-container">'
                f'<img class="preview-img" src="data:{mime_type};base64,{encoded_img}" style="max-height: 380px;"/>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Classify
            prediction_val = model.predict(processed_vector)[0]
            confidence = calculate_confidence(prediction_val)

            # Build card layout based on output
            if prediction_val >= 0.5:
                card_class = "result-card-male"
                badge_class = "gender-badge-male"
                gender = "Male"
                icon = "👨"
                bar_class = "progress-bar-fill-male"
            else:
                card_class = "result-card-female"
                badge_class = "gender-badge-female"
                gender = "Female"
                icon = "👩"
                bar_class = "progress-bar-fill-female"

            # Render Result Card
            st.markdown(
                f'<div class="{card_class}">'
                f'  <div class="{badge_class}">{icon} {gender}</div>'
                f'  <div class="progress-section">'
                f'    <div class="progress-label-container">'
                f'      <span>Confidence</span>'
                f'      <span>{confidence:.0f}%</span>'
                f'    </div>'
                f'    <div class="progress-bar-bg">'
                f'      <div class="{bar_class}" style="width: {confidence:.0f}%;"></div>'
                f'    </div>'
                f'  </div>'
                f'  <div class="metric-container">'
                f'    <div class="metric-box">'
                f'      <div class="metric-label">Prediction Score</div>'
                f'      <div class="metric-value">{prediction_val:.4f}</div>'
                f'    </div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Premium Footer
    st.markdown(
        '<div class="footer">'
        'Made with ❤️ using Python, Streamlit & Scikit-Learn'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
