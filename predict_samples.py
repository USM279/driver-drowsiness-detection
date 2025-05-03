import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model (custom or VGG16)
model = load_model('custom_cnn_model.h5')  # Change to vgg16_transfer_model.h5 if needed

# Image settings
img_size = (150, 150)
class_names = ['alert', 'drowsy']
base_path = 'data_split/test'

# Choose 3 random samples (from both classes)
samples = []
for cls in class_names:
    folder = os.path.join(base_path, cls)
    imgs = os.listdir(folder)
    selected = random.sample(imgs, 1)  # 1 from each class
    samples.extend([(os.path.join(folder, img), cls) for img in selected])

# Add 3rd image randomly
all_imgs = []
for cls in class_names:
    folder = os.path.join(base_path, cls)
    imgs = os.listdir(folder)
    all_imgs.extend([(os.path.join(folder, img), cls) for img in imgs])
samples.append(random.choice(all_imgs))

# Predict and show
for img_path, actual_label in samples:
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_label = 'alert' if prediction[0][0] < 0.5 else 'drowsy'

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
    plt.show()
