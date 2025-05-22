import streamlit as st
import cv2
import numpy as np
import pickle
from pathlib import Path
import tempfile
import os

# Load the trained SVM model
@st.cache_resource # Cache the model loading
def load_model():
    try:
        with open(Path('models/svm_model.pkl'), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Extract features function (same as in train_model.py and GUI)
def extract_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().reshape(1, -1)

# Classify frame function (adapted from GUI)
def classify_frame(frame, model, class_names):
    # Extract features from the frame
    features = extract_features(frame)
    
    # Get prediction
    prediction = model.predict(features)[0]
    confidence = model.decision_function(features)[0]
    
    # Add prediction text to frame
    label = f"{class_names[prediction]} ({abs(confidence):.2f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# --- Streamlit App --- 

st.title("Fire and Smoke Detection Web App")

model = load_model()
class_names = ['Fire', 'Smoke'] # Two classes for now based on our last setup

if model is None:
    st.warning("Trained model not found. Please run `python train_model.py` first.")
else:
    st.success("Model loaded successfully.")

uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    st.write("Processing video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
    else:
        stframe = st.empty() # Create an empty container for the video frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = classify_frame(frame, model, class_names)
            
            # Convert processed frame from BGR to RGB for Streamlit display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in Streamlit
            stframe.image(processed_frame_rgb, channels="RGB")
            
        cap.release()
        os.unlink(video_path) # Clean up the temporary file
        st.write("Video processing finished.") 