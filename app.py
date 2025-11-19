import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceApi

# ---------------------------------------
# Load environment variables
# ---------------------------------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# IMPORTANT:
# Use the correct Hugging Face INFERENCE API endpoint:
HF_API_URL = "https://router.huggingface.co/v1/models/google/medgemma-4b"

# ---------------------------------------
# Hugging Face text generation function
# ---------------------------------------
def generate_report(image_path):
    if HF_API_KEY is None:
        return "❌ ERROR: Hugging Face API key not found. Please check your .env file."

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"Analyze this chest X-ray image located at '{image_path}'. "
        "Provide a detailed medical-style interpretation of whether tuberculosis is present, "
        "what features support the conclusion, and any other clinical insights."
    )

    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return f"❌ Hugging Face API Error: {response.text}"

        result = response.json()

        # HuggingFace returns a list with "generated_text"
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]

        # Some models return { "generated_text": "..." }
        if "generated_text" in result:
            return result["generated_text"]

        return str(result)

    except Exception as e:
        return f"❌ API Request Failed: {e}"

# ---------------------------------------
# Image preprocessing
# ---------------------------------------
def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((512, 512))
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = tf.expand_dims(arr, axis=0)
    return arr

# ---------------------------------------
# Prediction function
# ---------------------------------------
def predict_tuberculosis(img):
    arr = preprocess_image(img)
    prediction = model.predict(arr)[0][0]   # Extract scalar
    return prediction

# ---------------------------------------
# Save uploaded images
# ---------------------------------------
def save_uploaded_image(img):
    folder = "uploaded_images"
    os.makedirs(folder, exist_ok=True)

    image_path = os.path.join(folder, f"image_{len(os.listdir(folder)) + 1}.png")
    img.save(image_path)

    return image_path

# ---------------------------------------
# Load your TB detection model
# ---------------------------------------
model = load_model("tb_detection_model.h5")

# ---------------------------------------
# Streamlit app UI
# ---------------------------------------
st.title("🩺 Tuberculosis Detection from Chest X-ray")
st.markdown("""
Upload a chest X-ray image.  
The system will:
1. Predict whether TB is present using your trained model  
2. Generate an explanatory report using a Hugging Face medical LLM
""")

# ---------------------------------------
# File upload
# ---------------------------------------
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # Save image locally
    image_path = save_uploaded_image(img)

    # Run model prediction
    probability = predict_tuberculosis(img)
    label = "Tuberculosis Detected" if probability >= 0.5 else "Normal"

    st.subheader("🔍 Prediction")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {probability:.4f}")

    # Generate HF report
    st.subheader("📝 AI-Generated Radiology Report")
    report = generate_report(image_path)
    st.write(report)

    st.info("Image successfully saved for further analysis.")
