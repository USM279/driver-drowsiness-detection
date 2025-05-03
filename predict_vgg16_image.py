import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('vgg16_transfer_model.h5')  # or 'custom_cnn_model.h5'

# Image settings
img_size = (150, 150)
class_names = ['alert', 'drowsy']
base_path = 'data_split/test'

# Load 1 image from alert and 1 from drowsy
samples = []
for cls in class_names:
    folder = os.path.join(base_path, cls)
    img_name = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, img_name)
    samples.append((img_path, cls))

# Add personal image
samples.append(('external/test_image.jpg', 'personal'))

# Predict and display
for img_path, actual_label in samples:
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_label = 'alert' if prediction[0][0] < 0.5 else 'drowsy'

    title = f"Actual: {actual_label} | Predicted: {predicted_label}" if actual_label != 'personal' else f"Personal Image | Predicted: {predicted_label}"

    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()
