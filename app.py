import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.title("Driver Drowsiness Detection")
model = load_model('models/vgg16_transfer_model.h5')

uploaded_file = st.file_uploader("Upload a facial image", type=['jpg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)[0][0]

    label = "Drowsy 😴" if pred > 0.5 else "Alert 🚗"
    confidence = f"{(pred if pred > 0.5 else 1 - pred) * 100:.2f}%"

    st.success(f"Prediction: {label}")
    st.info(f"Model Confidence: {confidence}")
