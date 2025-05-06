import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Function to preprocess ECG data
def preprocess_data(X, expected_features=187):
    # Check number of features
    if X.shape[1] < expected_features:
        st.error(f"CSV has {X.shape[1]} feature columns, but model expects {expected_features}. Please provide a valid CSV.")
        return None
    if X.shape[1] > expected_features:
        st.warning(f"CSV has {X.shape[1]} feature columns, truncating to {expected_features}.")
        X = X[:, :expected_features]
    
    # Normalize features
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-7)
    X = X.astype(np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X

# Streamlit app
def streamlit_app():
    st.set_page_config(page_title="ECG Classification App", page_icon="❤️", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main { background: linear-gradient(to right, #e0e7ff, #f3e8ff); padding: 20px; }
        .stButton>button { 
            background: linear-gradient(to right, #4f46e5, #7c3aed); 
            color: white; 
            border-radius: 9999px; 
            padding: 12px 24px; 
            font-weight: 600;
        }
        .stButton>button:hover { 
            background: linear-gradient(to right, #4338ca, #6d28d9); 
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("ECG Signal Classification")
    st.write("Upload a CNN-LSTM model (.keras) and a test CSV to classify ECG signals into classes: N, S, V, F, Q.")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload Model Checkpoint (.keras)", type=["keras"])
    with col2:
        test_file = st.file_uploader("Upload Test Dataset (.csv)", type=["csv"])

    if st.button("Classify ECG Signals", disabled=not (model_file and test_file)):
        with st.spinner("Processing and classifying..."):
            # Save files temporarily
            model_path = f"temp_{model_file.name}"
            test_path = f"temp_{test_file.name}"
            with open(model_path, "wb") as f:
                f.write(model_file.read())
            with open(test_path, "wb") as f:
                f.write(test_file.read())

            # Load test data
            try:
                test_data = pd.read_csv(test_path, header=None)
                X_test = test_data.iloc[:, :-1].values  # Ignore last column (labels)
            except Exception as e:
                st.error(f"Error reading CSV: {e}. Ensure it's a valid CSV with at least 187 feature columns.")
                os.remove(model_path)
                os.remove(test_path)
                return

            # Debug: Show CSV info
            st.write("### CSV Info")
            st.write(f"Number of samples: {X_test.shape[0]}")
            st.write(f"Number of feature columns: {X_test.shape[1]}")
            st.write("Sample of first row (first 5 values):", X_test[0, :5])

            # Preprocess data
            X_test = preprocess_data(X_test, expected_features=187)
            if X_test is None:
                os.remove(model_path)
                os.remove(test_path)
                return

            # Load model
            try:
                model = tf.keras.models.load_model(model_path)
            except Exception as e:
                st.error(f"Error loading model: {e}. Ensure it's a valid .keras model expecting input shape (None, 187, 1).")
                os.remove(model_path)
                os.remove(test_path)
                return

            # Classify
            try:
                y_pred = model.predict(X_test, verbose=0)
                y_pred_cls = np.argmax(y_pred, axis=1)
                class_names = ['N', 'S', 'V', 'F', 'Q']
                y_pred_labels = [class_names[i] for i in y_pred_cls]
            except Exception as e:
                st.error(f"Error during prediction: {e}. Ensure CSV features match model input (187 features).")
                os.remove(model_path)
                os.remove(test_path)
                return

            # Clean up
            os.remove(model_path)
            os.remove(test_path)

            # Display results
            st.success("Classification completed!")
            st.subheader("Predicted Classes")
            results_df = pd.DataFrame({
                "Sample ID": range(1, len(y_pred_labels) + 1),
                "Predicted Class": y_pred_labels
            })
            st.dataframe(results_df, use_container_width=True)

            # Optional: Show class distribution
            st.subheader("Class Distribution")
            class_counts = pd.Series(y_pred_labels).value_counts()
            st.bar_chart(class_counts)

# Run app
if __name__ == "__main__":
    streamlit_app()