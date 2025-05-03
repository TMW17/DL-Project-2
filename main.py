import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pywt
import streamlit as st
import streamlit_shadcn_ui as ui
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="ECG Arrhythmia Classifier", layout="wide")

# Define model path
MODEL_PATH = "/Users/tadjmohamedwalid/Desktop/Project DL/kaggle/working/best_ecg_model.keras"

# Load the trained model
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please ensure the file exists.")
        return None
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# Define label mapping
LABEL_MAP = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    selected_tab = ui.tabs(options=["Home", "Predict", "About"], default_value="Home", key="tabs")

# Home tab
if selected_tab == "Home":
    st.title("ECG Arrhythmia Classifier")
    st.markdown("""
    Welcome to the ECG Arrhythmia Classifier application. This tool allows you to classify ECG signals into different arrhythmia categories using a trained deep learning model.

    **Features:**
    - Upload ECG signals for classification.
    - Visualize the ECG waveform.
    - View classification results and probabilities.

    👉 To test it quickly, download and upload this [sample ECG file](https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0/100.csv)
    """)

# Predict tab
elif selected_tab == "Predict":
    st.title("Upload ECG Signal for Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file containing ECG signal", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            signal = pd.read_csv(uploaded_file, header=None).values.flatten().astype(np.float32)

            # Define signal window
            WINDOW_SIZE = 200
            HALF_WINDOW = WINDOW_SIZE // 2

            if len(signal) < WINDOW_SIZE:
                st.error("The uploaded signal is too short. Please upload a signal with at least 200 data points.")
            else:
                # Extract a central segment
                center = len(signal) // 2
                segment = signal[center - HALF_WINDOW:center + HALF_WINDOW]

                # Wavelet denoising
                coeffs = pywt.wavedec(segment, 'db6', level=4)
                coeffs[-1] = np.zeros_like(coeffs[-1])
                coeffs[-2] = np.zeros_like(coeffs[-2])
                denoised = pywt.waverec(coeffs, 'db6')[:WINDOW_SIZE]

                # Normalize
                normalized = (denoised - np.mean(denoised)) / (np.std(denoised) + 1e-7)
                input_signal = normalized.reshape(1, WINDOW_SIZE, 1)

                # Predict
                prediction = model.predict(input_signal)
                predicted_label = INV_LABEL_MAP[np.argmax(prediction)]

                # Results
                st.subheader("Prediction Results")
                st.write(f"**Predicted Label:** {predicted_label}")
                st.write("**Prediction Probabilities:**")
                for label, prob in zip(LABEL_MAP.keys(), prediction[0]):
                    st.write(f"{label}: {prob:.4f}")

                # ECG Plot
                st.subheader("ECG Signal")
                fig, ax = plt.subplots()
                ax.plot(segment)
                ax.set_title("ECG Signal Segment")
                ax.set_xlabel("Time")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# About tab
elif selected_tab == "About":
    st.title("About This Application")
    st.markdown("""
    This application is developed to classify ECG signals into various arrhythmia categories using a deep learning model trained on the MIT-BIH Arrhythmia Database.

    **Model Details:**
    - Architecture: Residual CNN with Separable Convolutions
    - Input: 200-sample ECG segments
    - Output: 5 arrhythmia classes

    **Developer:**
    - Name: Tadj Mohamed Walid
    - Location: Le Lido, Alger, Algeria
    """)
