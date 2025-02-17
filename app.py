import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .main { background-color: #f5f5f5; }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .prediction-box {
            padding: 15px;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 8px 0;
        }
        .probability-container {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .stProgress { height: 20px; border-radius: 10px; }
        .stProgress > div > div { background-color: #0066cc; }
    </style>
""", unsafe_allow_html=True)

def validate_file_size(file):
    """
    Validate if file size is under 5MB
    """
    MAX_SIZE = 5 * 1024 * 1024  # 5MB in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size <= MAX_SIZE

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    return preprocess_input(img), img

def main():
    st.title("👁 Eye Disease Classification")
    
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('interface/model.keras')
    
    try:
        model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        
        if not validate_file_size(uploaded_file):
            st.error("❌ File size must be under 5MB")
            return
        
        try:
            
            image = Image.open(uploaded_file)
            
            if st.button("🔍 Analyze Image"):
                with st.spinner("Processing image..."):
                    preprocessed, resized = preprocess_image(image)
                    
                    st.markdown("### Image Processing Pipeline")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("*Original Image*")
                        st.image(image, width=300)
                    
                    with col2:
                        st.markdown("*Preprocessed Image (224x224)*")
                        st.image(resized, width=300)
                    
                    prediction = model.predict(np.expand_dims(preprocessed, axis=0))
                    classes = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract']
                    
                    st.markdown("### Analysis Results")
                    max_prob_idx = np.argmax(prediction[0])
                    
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h3 style='text-align: center; color: #0066cc;'>Primary Diagnosis</h3>
                            <h2 style='text-align: center;'>{classes[max_prob_idx]}</h2>
                            <h4 style='text-align: center;'>Confidence: {prediction[0][max_prob_idx]*100:.1f}%</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    col_chart1, col_chart2 = st.columns([3, 2])
                    
                    with col_chart1:
                        for class_name, prob in zip(classes, prediction[0]):
                            st.markdown(f"<div class='probability-container'>", unsafe_allow_html=True)
                            st.markdown(f"{class_name}")
                            st.progress(float(prob))
                            st.markdown(f"Probability: {prob*100:.1f}%")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_chart2:
                        if max_prob_idx == 0:
                            st.success("✅ No significant eye conditions detected.")
                            st.markdown("""
                                *Recommendations:*
                                - Continue regular eye check-ups
                                - Maintain good eye care habits
                            """)
                        else:
                            st.warning("⚠ Potential eye condition detected")
                            st.markdown("""
                                *Important Notice:*
                                - This is an AI-assisted diagnosis
                                - Please consult an ophthalmologist
                                - Regular monitoring is recommended
                            """)
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            return                    

if __name__ == '__main__':
   main()