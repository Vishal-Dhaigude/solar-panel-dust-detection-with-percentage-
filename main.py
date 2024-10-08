import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
@st.cache_resource
def load_my_model():
    model = load_model('D:\my work and entertainment\mini sem 5 own\dust_detection_model.h5')  # Update with your model path
    return model

# Load the model
model = load_my_model()

# Streamlit app title
st.title("Solar Panel Dust Detection")
st.header("Upload an image of a solar panel to check for dust")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Adjust according to your model input size
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_names = ['Clean', 'Dusty']  # Update with your class names
    confidence = np.max(predictions[0]) * 100

    # Show results
    st.write(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")

# Optionally, show model summary
if st.checkbox("Show Model Summary"):
    st.text(model.summary())
