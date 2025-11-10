import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
import os
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Access the Hugging Face API key from the environment
HF_API_KEY = os.getenv('HF_API_KEY')  # Fetch the API key securely from .env
HF_API_URL = "https://huggingface.co/google/medgemma-4b-it"  # Update with your model's URL

# Function to interact with Hugging Face API
def generate_report(image_path):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {
        "inputs": f"Analyze the tuberculosis scan and provide detailed description for the image at {image_path}"
    }
    response = requests.post(HF_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['generated_text']
    else:
        return "Error generating the report. Please try again."

# Function to preprocess the uploaded image for prediction
def preprocess_image(img):
    # Convert grayscale to RGB by duplicating the grayscale channel to 3 channels
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert to RGB (3 channels)
        
    img = img.resize((512, 512))  # Ensure the image size is 512x512
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize the image
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make prediction using the loaded model
def predict_tuberculosis(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return prediction

# Function to save the uploaded image for further use
def save_uploaded_image(img):
    if not os.path.exists("uploaded_images"):
        os.makedirs("uploaded_images")
    image_path = os.path.join("uploaded_images", f"image_{len(os.listdir('uploaded_images')) + 1}.png")
    img.save(image_path)
    return image_path

# Load the pre-trained model
model = load_model('tb_detection_model.h5')

# Streamlit app UI
st.title("Tuberculosis Detection using Chest X-ray")

st.markdown("""
    This application allows you to upload a chest X-ray image to detect tuberculosis.
    The system will also generate a detailed description of the detected issue using a Hugging Face API.
    """)

# Image upload section
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image
    image_path = save_uploaded_image(img)

    # Make predictions
    prediction = predict_tuberculosis(img)
    prediction_label = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
    st.write(f"Prediction: {prediction_label}")

    # Generate report from Hugging Face API
    report = generate_report(image_path)
    st.subheader("Detailed Report:")
    st.write(report)

    # Optionally: You can implement a mechanism to store the image and use it for further training.
    st.write("The image has been saved for further use.")
