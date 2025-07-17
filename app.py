import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image # Pillow for image processing
import os

# --- Configuration ---
MODEL_PATH = 'fashion_mnist_cnn_model.h5' # <--- CHANGE THIS
IMG_HEIGHT = 28 # <--- CHANGE THIS
IMG_WIDTH = 28 # <--- CHANGE THIS
IMG_CHANNELS = 1 # <--- CHANGE THIS (Grayscale)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Model Loading (with Streamlit caching for performance) ---
@st.cache_resource # Cache the model loading to prevent reloading on every interaction
def load_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model not found at {MODEL_PATH}")
        st.error("Please ensure you have trained and saved the Fashion MNIST CNN model.")
        st.stop() # Stop the app if model is not found
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model once
model = load_model()

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """
    Preprocesses a PIL Image for the simple Fashion MNIST CNN model prediction.
    - Converts to Grayscale (1 channel).
    - Resizes to target dimensions (28x28).
    - Converts to a float32 tensor in [0, 1] range.
    - Adds a batch dimension.
    """
    # Convert to grayscale (1 channel)
    image = image.convert('L') # 'L' mode for grayscale

    # Resize image to 28x28
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixel values to [0, 1] (Fashion MNIST typically uses this)
    image_array = image_array / 255.0

    # Convert to TensorFlow tensor and add channel dimension for grayscale (28, 28) -> (28, 28, 1)
    # And then add batch dimension (28, 28, 1) -> (1, 28, 28, 1)
    image_tensor = tf.expand_dims(image_array, axis=-1) # Add channel dimension
    image_tensor = tf.expand_dims(image_tensor, axis=0) # Add batch dimension
    
    # Ensure dtype is float32
    image_tensor = tf.cast(image_tensor, tf.float32)

    return image_tensor

# --- Streamlit UI (No changes needed here, as it calls the above functions) ---
st.set_page_config(
    page_title="Fashion Item Classifier (Simple CNN)", # <--- Optional: update title
    page_icon="👕",
    layout="centered"
)

st.title("👕 Fashion Item Classifier (Simple CNN)") # <--- Optional: update title
st.markdown("Upload an image of a clothing item, and I'll try to classify it using a simple CNN model trained on Fashion MNIST!")
st.info(f"Model: Simple CNN. Input size: {IMG_HEIGHT}x{IMG_WIDTH} Grayscale.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.write("")
        st.write("### Making Prediction...")
        
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction_probabilities = model.predict(processed_image)[0]
        predicted_class_index = np.argmax(prediction_probabilities)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction_probabilities[predicted_class_index] * 100

        st.success(f"**Predicted Class:** {predicted_class_name}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Display all probabilities
        st.write("#### All Class Probabilities:")
        prob_df = tf.DataFrame({
            "Class": class_names,
            "Probability": prediction_probabilities
        }).sort_values(by="Probability", ascending=False)
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

        st.warning("💡 **Note:** This model is a simple CNN trained on 28x28 grayscale Fashion MNIST images. It may not perform well on complex real-world photos due to resolution, color, background, and pose variations.")

# Instructions for running the app
st.markdown("""
---
### How to Use:
1.  **Save this code** as `app.py`.
2.  **Ensure your trained model** (`fashion_mnist_cnn_model.h5`) is in a `models` folder in the same directory as `app.py`.
3.  **Open your terminal**, navigate to this directory.
4.  **Run the Streamlit app:** `streamlit run app.py`
5.  Upload an image of a T-shirt, trouser, shoe, or any other Fashion MNIST category!
""")