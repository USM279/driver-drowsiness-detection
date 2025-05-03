import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to your image
img_path = 'external/test_image.jpg'  # Change this if needed
img_size = (150, 150)

# Load the model
model = load_model('custom_cnn_model.h5')

# Load and prepare the image
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
predicted_label = 'alert' if prediction[0][0] < 0.5 else 'drowsy'

# Show result
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted: {predicted_label}")
plt.show()
